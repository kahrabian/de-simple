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

    def t_emb(self, e, y, m, d):
        y_emb = self.y_amp(e) * self.t_nl(self.y_frq(e) * y + self.y_phi(e))
        m_emb = self.m_amp(e) * self.t_nl(self.m_frq(e) * m + self.m_phi(e))
        d_emb = self.d_amp(e) * self.t_nl(self.d_frq(e) * d + self.d_phi(e))

        return y_emb + m_emb + d_emb

    def emb(self, s, r, o, y, m, d):
        y, m, d = y.view(-1, 1), m.view(-1, 1), d.view(-1, 1)

        s_emb = T.cat((self.e_emb(s), self.t_emb(s, y, m, d)), 1)
        o_emb = T.cat((self.e_emb(o), self.t_emb(o, y, m, d)), 1)
        r_emb = self.r_emb(r)

        return s_emb, r_emb, o_emb

    def _r_emb(self, s_t, s_r, s_e, o_t, o_r, o_e):
        s_re = list(map(lambda x: T.cat([self.r_emb(x[0].flip(0)), self.e_emb(x[1].flip(0))], dim=1), zip(s_r, s_e)))
        o_re = list(map(lambda x: T.cat([self.r_emb(x[0].flip(0)), self.e_emb(x[1].flip(0))], dim=1), zip(o_r, o_e)))

        r_s = pad_sequence(s_re).flip(0)
        r_o = pad_sequence(o_re).flip(0)

        m_s = None
        for r_s_chk in T.chunk(r_s, int(np.ceil(r_s.size(0) / self.ql)), 0):
            h_s, m_s = self.mt(r_s_chk, m_s)

        m_o = None
        for r_o_chk in T.chunk(r_o, int(np.ceil(r_o.size(0) / self.ql)), 0):
            h_o, m_o = self.mt(r_o_chk, m_o)

        return h_s[-1], h_o[-1]

    def forward(self, s, r, o, y, m, d, s_t, s_r, s_e, o_t, o_r, o_e):
        s_emb, r_emb, o_emb = self.emb(s, r, o, y, m, d)

        sc = (s_emb * r_emb) * o_emb
        sc = F.dropout(sc, p=self.drp, training=self.training)
        sc = T.sum(sc, dim=1)

        s_r_emb, o_r_emb = self._r_emb(s_t, s_r, s_e, o_t, o_r, o_e)

        sc_r = s_r_emb * o_r_emb
        sc_r = F.dropout(sc_r, p=self.drp, training=self.training)
        sc_r = T.sum(sc_r, dim=1)

        return sc
