import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from src.models.transformers import MemTransformerXL


class DESimplE(nn.Module):
    def __init__(self, ne, nr, args):
        super(DESimplE, self).__init__()
        self.drp = args.drp
        self.ql = args.ql

        self.e_emb_s = nn.Embedding(ne, args.s_dim).to(args.dvc)
        self.e_emb_o = nn.Embedding(ne, args.s_dim).to(args.dvc)
        self.r_emb_f = nn.Embedding(nr, args.s_dim + args.t_dim).to(args.dvc)
        self.r_emb_i = nn.Embedding(nr, args.s_dim + args.t_dim).to(args.dvc)
        nn.init.xavier_uniform_(self.e_emb_s.weight)
        nn.init.xavier_uniform_(self.e_emb_o.weight)
        nn.init.xavier_uniform_(self.r_emb_f.weight)
        nn.init.xavier_uniform_(self.r_emb_i.weight)

        self.m_frq_s = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.m_frq_o = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.d_frq_s = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.d_frq_o = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.y_frq_s = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.y_frq_o = nn.Embedding(ne, args.t_dim).to(args.dvc)
        nn.init.xavier_uniform_(self.m_frq_s.weight)
        nn.init.xavier_uniform_(self.d_frq_s.weight)
        nn.init.xavier_uniform_(self.y_frq_s.weight)
        nn.init.xavier_uniform_(self.m_frq_o.weight)
        nn.init.xavier_uniform_(self.d_frq_o.weight)
        nn.init.xavier_uniform_(self.y_frq_o.weight)

        self.m_phi_s = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.m_phi_o = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.d_phi_s = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.d_phi_o = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.y_phi_s = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.y_phi_o = nn.Embedding(ne, args.t_dim).to(args.dvc)
        nn.init.xavier_uniform_(self.m_phi_s.weight)
        nn.init.xavier_uniform_(self.d_phi_s.weight)
        nn.init.xavier_uniform_(self.y_phi_s.weight)
        nn.init.xavier_uniform_(self.m_phi_o.weight)
        nn.init.xavier_uniform_(self.d_phi_o.weight)
        nn.init.xavier_uniform_(self.y_phi_o.weight)

        self.m_amp_s = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.m_amp_o = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.d_amp_s = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.d_amp_o = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.y_amp_s = nn.Embedding(ne, args.t_dim).to(args.dvc)
        self.y_amp_o = nn.Embedding(ne, args.t_dim).to(args.dvc)
        nn.init.xavier_uniform_(self.m_amp_s.weight)
        nn.init.xavier_uniform_(self.d_amp_s.weight)
        nn.init.xavier_uniform_(self.y_amp_s.weight)
        nn.init.xavier_uniform_(self.m_amp_o.weight)
        nn.init.xavier_uniform_(self.d_amp_o.weight)
        nn.init.xavier_uniform_(self.y_amp_o.weight)

        self.t_nl = T.sin

        self.mt = MemTransformerXL(n_l=args.nl, n_h=args.nh,
                                   d_in=args.s_dim * 2 + args.t_dim,
                                   d_h=args.s_dim * 2 + args.t_dim,
                                   d_hid=args.s_dim * 2 + args.t_dim,
                                   drp=self.drp, drp_att=self.drp,
                                   pl_nrm=True, m_l=args.ml)

    def t_emb_s(self, e, y, m, d):
        y_emb = self.y_amp_s(e) * self.t_nl(self.y_frq_s(e) * y + self.y_phi_s(e))
        m_emb = self.m_amp_s(e) * self.t_nl(self.m_frq_s(e) * m + self.m_phi_s(e))
        d_emb = self.d_amp_s(e) * self.t_nl(self.d_frq_s(e) * d + self.d_phi_s(e))

        return y_emb + m_emb + d_emb

    def t_emb_o(self, e, y, m, d):
        y_emb = self.y_amp_o(e) * self.t_nl(self.y_frq_o(e) * y + self.y_phi_o(e))
        m_emb = self.m_amp_o(e) * self.t_nl(self.m_frq_o(e) * m + self.m_phi_o(e))
        d_emb = self.d_amp_o(e) * self.t_nl(self.d_frq_o(e) * d + self.d_phi_o(e))

        return y_emb + m_emb + d_emb

    def emb(self, s, r, o, y, m, d):
        y, m, d = y.view(-1, 1), m.view(-1, 1), d.view(-1, 1)

        s_emb_s = T.cat((self.e_emb_s(s), self.t_emb_s(s, y, m, d)), 1)
        o_emb_o = T.cat((self.e_emb_o(o), self.t_emb_o(o, y, m, d)), 1)
        r_emb_f = self.r_emb_f(r)

        o_emb_s = T.cat((self.e_emb_s(o), self.t_emb_s(o, y, m, d)), 1)
        s_emb_o = T.cat((self.e_emb_o(s), self.t_emb_o(s, y, m, d)), 1)
        r_emb_i = self.r_emb_i(r)

        return s_emb_s, r_emb_f, o_emb_o, o_emb_s, r_emb_i, s_emb_o

    def _r_emb(self, s_t, s_r, s_e, o_t, o_r, o_e):
        s_s_re = list(map(lambda x: T.cat([self.r_emb_f(x[0].flip(0)),
                                           self.e_emb_o(x[1].flip(0))], dim=1), zip(s_r, s_e)))
        o_o_re = list(map(lambda x: T.cat([self.r_emb_i(x[0].flip(0)),
                                           self.e_emb_s(x[1].flip(0))], dim=1), zip(o_r, o_e)))

        r_s_s = pad_sequence(s_s_re).flip(0)
        r_o_o = pad_sequence(o_o_re).flip(0)

        s_o_se = list(map(lambda x: T.cat([self.r_emb_f(x[0].flip(0)),
                                           self.e_emb_o(x[1].flip(0))], dim=1), zip(o_r, o_e)))
        o_s_se = list(map(lambda x: T.cat([self.r_emb_i(x[0].flip(0)),
                                           self.e_emb_s(x[1].flip(0))], dim=1), zip(s_r, s_e)))

        r_s_o = pad_sequence(s_o_se).flip(0)
        r_o_s = pad_sequence(o_s_se).flip(0)

        m_s_s = None
        for r_s_s_chk in T.chunk(r_s_s, int(np.ceil(r_s_s.size(0) / self.ql)), 0):
            h_s_s, m_s_s = self.mt(r_s_s_chk, m_s_s)

        m_o_o = None
        for r_o_o_chk in T.chunk(r_o_o, int(np.ceil(r_o_o.size(0) / self.ql)), 0):
            h_o_o, m_o_o = self.mt(r_o_o_chk, m_o_o)

        m_s_o = None
        for r_s_o_chk in T.chunk(r_s_o, int(np.ceil(r_s_o.size(0) / self.ql)), 0):
            h_s_o, m_s_o = self.mt(r_s_o_chk, m_s_o)

        m_o_s = None
        for r_o_s_chk in T.chunk(r_o_s, int(np.ceil(r_o_s.size(0) / self.ql)), 0):
            h_o_s, m_o_s = self.mt(r_o_s_chk, m_o_s)

        return h_s_s[-1], h_o_o[-1], h_s_o[-1], h_o_s[-1]

    def forward(self, s, r, o, y, m, d, s_t, s_r, s_e, o_t, o_r, o_e):
        s_emb_s, r_emb_f, o_emb_o, o_emb_s, r_emb_i, s_emb_o = self.emb(s, r, o, y, m, d)

        sc = ((s_emb_s * r_emb_f) * o_emb_o + (o_emb_s * r_emb_i) * s_emb_o) / 2.0
        sc = F.dropout(sc, p=self.drp, training=self.training)
        sc = T.sum(sc, dim=1)

        s_s_r_emb, o_o_r_emb, s_o_r_emb, o_s_r_emb = self._r_emb(s_t, s_r, s_e, o_t, o_r, o_e)

        sc_r = ((s_s_r_emb * o_o_r_emb) + (s_o_r_emb * o_s_r_emb)) / 2.0
        sc_r = F.dropout(sc_r, p=self.drp, training=self.training)
        sc_r = (-1) * T.norm(sc_r, dim=1)

        return sc
