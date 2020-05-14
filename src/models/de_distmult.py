import pytz
from datetime import datetime

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from src.models.transformers import MemTransformerXL


class DEDistMult(nn.Module):
    def __init__(self, ne, nr, args):
        super(DEDistMult, self).__init__()
        self.drp = args.drp
        self.ql = args.ql
        self.dvc = args.dvc

        self.e_emb = nn.Embedding(ne, args.s_dim).to(args.dvc)
        self.r_emb = nn.Embedding(nr, args.s_dim + args.t_dim).to(args.dvc)
        nn.init.xavier_uniform_(self.e_emb.weight)
        nn.init.xavier_uniform_(self.r_emb.weight)

        self.m_frq = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.d_frq = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.y_frq = nn.Embedding(ne, args.t_dim).to(args.dvc)
        nn.init.xavier_uniform_(self.m_frq.weight)
        nn.init.xavier_uniform_(self.d_frq.weight)
        nn.init.xavier_uniform_(self.y_frq.weight)

        self.m_phi = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.d_phi = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.y_phi = nn.Embedding(ne, args.t_dim).to(args.dvc)
        nn.init.xavier_uniform_(self.m_phi.weight)
        nn.init.xavier_uniform_(self.d_phi.weight)
        nn.init.xavier_uniform_(self.y_phi.weight)

        self.m_amp = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.d_amp = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.y_amp = nn.Embedding(ne, args.t_dim).to(args.dvc)
        nn.init.xavier_uniform_(self.m_amp.weight)
        nn.init.xavier_uniform_(self.d_amp.weight)
        nn.init.xavier_uniform_(self.y_amp.weight)

        self.t_nl = T.sin

        self.mt = MemTransformerXL(n_l=args.nl, n_h=args.nh,
                                   d_in=args.s_dim * 2 + args.t_dim,
                                   d_h=args.s_dim * 2 + args.t_dim,
                                   d_hid=args.s_dim * 2 + args.t_dim,
                                   drp=self.drp, drp_att=self.drp,
                                   pl_nrm=True, m_l=args.ml)

    def _to(self, *t):
        return list(map(lambda x: x.to(self.dvc), t))

    def t_emb(self, e, y, m, d):
        y_emb = self.y_amp(e) * self.t_nl(self.y_frq(e) * y + self.y_phi(e))
        m_emb = self.m_amp(e) * self.t_nl(self.m_frq(e) * m + self.m_phi(e))
        d_emb = self.d_amp(e) * self.t_nl(self.d_frq(e) * d + self.d_phi(e))

        return y_emb + m_emb + d_emb

    def emb(self, s, r, o, y, m, d):
        s, r, o, y, m, d = self._to(s, r, o, y, m, d)

        y, m, d = y.view(-1, 1), m.view(-1, 1), d.view(-1, 1)

        s_emb = T.cat((self.e_emb(s), self.t_emb(s, y, m, d)), 1)
        o_emb = T.cat((self.e_emb(o), self.t_emb(o, y, m, d)), 1)
        r_emb = self.r_emb(r)

        return s_emb, r_emb, o_emb

    def _transformer(self, e_t, t, r, e):
        r_t = pad_sequence(list(map(lambda x: x[0] - x[1].flip(0), zip(e_t, t)))).flip(0)
        r_re = list(map(lambda x: T.cat([self.r_emb(x[0].flip(0)), self.e_emb(x[1].flip(0))], dim=1), zip(r, e)))
        r_re = pad_sequence(r_re).flip(0)

        m = None
        chk_cnt = int(np.ceil(r_re.size(0) / self.ql))
        for re_chk, t_chk in zip(T.chunk(r_re, chk_cnt, 0), T.chunk(r_t, chk_cnt, 0)):
            h, m = self.mt(re_chk.to(self.dvc), t_chk.to(self.dvc), m)
        return h[-1]

    def _r_emb(self, s_t, s_r, s_e, o_t, o_r, o_e, y, m, d):
        y_l, m_l, d_l = y.long(), m.long(), d.long()
        t = [datetime(year=x[0], month=x[1], day=x[2], tzinfo=pytz.utc).toordinal() for x in zip(y_l, m_l, d_l)]
        return self._transformer(t, s_t, s_r, s_e), self._transformer(t, o_t, o_r, o_e)

    def forward(self, s, r, o, y, m, d, s_t, s_r, s_e, o_t, o_r, o_e):
        s_emb, r_emb, o_emb = self.emb(s, r, o, y, m, d)

        sc = (s_emb * r_emb) * o_emb
        sc = F.dropout(sc, p=self.drp, training=self.training)
        sc = T.sum(sc, dim=1)

        s_r_emb, o_r_emb = self._r_emb(s_t, s_r, s_e, o_t, o_r, o_e, y, m, d)

        sc_r = s_r_emb * o_r_emb
        sc_r = F.dropout(sc_r, p=self.drp, training=self.training)
        sc_r = T.sum(sc_r, dim=1)

        return sc
