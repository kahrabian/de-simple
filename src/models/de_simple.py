import torch
import torch.nn as nn
import torch.nn.functional as F


class DESimplE(torch.nn.Module):
    def __init__(self, ne, nr, args):
        super(DESimplE, self).__init__()
        self.drp = args.drp

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

        self.t_nl = torch.sin

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

        s_emb_s = torch.cat((self.e_emb_s(s), self.t_emb_s(s, y, m, d)), 1)
        o_emb_o = torch.cat((self.e_emb_o(o), self.t_emb_o(o, y, m, d)), 1)
        r_emb_f = self.r_emb_f(r)

        o_emb_s = torch.cat((self.e_emb_s(o), self.t_emb_s(o, y, m, d)), 1)
        s_emb_o = torch.cat((self.e_emb_o(s), self.t_emb_o(s, y, m, d)), 1)
        r_emb_i = self.r_emb_i(r)

        return s_emb_s, r_emb_f, o_emb_o, o_emb_s, r_emb_i, s_emb_o

    def forward(self, s, r, o, y, m, d):
        s_emb_s, r_emb_f, o_emb_o, o_emb_s, r_emb_i, s_emb_o = self.emb(s, r, o, y, m, d)

        sc = ((s_emb_s * r_emb_f) * o_emb_o + (o_emb_s * r_emb_i) * s_emb_o) / 2.0
        sc = F.dropout(sc, p=self.drp, training=self.training)
        sc = torch.sum(sc, dim=1)

        return sc
