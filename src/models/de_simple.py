import torch as T
import torch.nn as nn
import torch.nn.functional as F

from src.models.embedding import PositionalEmbedding


class DESimplE(nn.Module):
    def __init__(self, ne, nr, args):
        super(DESimplE, self).__init__()
        self.drp = args.drp
        self.t_dim = args.t_dim
        self.r_dim = args.r_dim

        self.e_emb_s = nn.Embedding(ne, args.s_dim)
        self.e_emb_o = nn.Embedding(ne, args.s_dim)
        self.r_emb_f = nn.Embedding(nr, args.s_dim + args.t_dim)
        self.r_emb_i = nn.Embedding(nr, args.s_dim + args.t_dim)
        nn.init.xavier_uniform_(self.e_emb_s.weight)
        nn.init.xavier_uniform_(self.e_emb_o.weight)
        nn.init.xavier_uniform_(self.r_emb_f.weight)
        nn.init.xavier_uniform_(self.r_emb_i.weight)

        if self.t_dim > 0:
            self.m_frq_s = nn.Embedding(ne, args.t_dim)
            self.m_frq_o = nn.Embedding(ne, args.t_dim)
            self.d_frq_s = nn.Embedding(ne, args.t_dim)
            self.d_frq_o = nn.Embedding(ne, args.t_dim)
            self.y_frq_s = nn.Embedding(ne, args.t_dim)
            self.y_frq_o = nn.Embedding(ne, args.t_dim)
            nn.init.xavier_uniform_(self.m_frq_s.weight)
            nn.init.xavier_uniform_(self.d_frq_s.weight)
            nn.init.xavier_uniform_(self.y_frq_s.weight)
            nn.init.xavier_uniform_(self.m_frq_o.weight)
            nn.init.xavier_uniform_(self.d_frq_o.weight)
            nn.init.xavier_uniform_(self.y_frq_o.weight)

            self.m_phi_s = nn.Embedding(ne, args.t_dim)
            self.m_phi_o = nn.Embedding(ne, args.t_dim)
            self.d_phi_s = nn.Embedding(ne, args.t_dim)
            self.d_phi_o = nn.Embedding(ne, args.t_dim)
            self.y_phi_s = nn.Embedding(ne, args.t_dim)
            self.y_phi_o = nn.Embedding(ne, args.t_dim)
            nn.init.xavier_uniform_(self.m_phi_s.weight)
            nn.init.xavier_uniform_(self.d_phi_s.weight)
            nn.init.xavier_uniform_(self.y_phi_s.weight)
            nn.init.xavier_uniform_(self.m_phi_o.weight)
            nn.init.xavier_uniform_(self.d_phi_o.weight)
            nn.init.xavier_uniform_(self.y_phi_o.weight)

            self.m_amp_s = nn.Embedding(ne, args.t_dim)
            self.m_amp_o = nn.Embedding(ne, args.t_dim)
            self.d_amp_s = nn.Embedding(ne, args.t_dim)
            self.d_amp_o = nn.Embedding(ne, args.t_dim)
            self.y_amp_s = nn.Embedding(ne, args.t_dim)
            self.y_amp_o = nn.Embedding(ne, args.t_dim)
            nn.init.xavier_uniform_(self.m_amp_s.weight)
            nn.init.xavier_uniform_(self.d_amp_s.weight)
            nn.init.xavier_uniform_(self.y_amp_s.weight)
            nn.init.xavier_uniform_(self.m_amp_o.weight)
            nn.init.xavier_uniform_(self.d_amp_o.weight)
            nn.init.xavier_uniform_(self.y_amp_o.weight)

            self.t_nl = T.sin

        if self.r_dim > 0:
            self.p_emb = PositionalEmbedding(self.r_dim)

            self.w_e_s = nn.Parameter(T.zeros(args.s_dim, self.r_dim))
            self.w_e_o = nn.Parameter(T.zeros(args.s_dim, self.r_dim))
            self.w_rp_f = nn.Parameter(T.zeros(nr, nr, 1))
            self.w_rp_i = nn.Parameter(T.zeros(nr, nr, 1))
            self.w_p_s = nn.Parameter(T.zeros(self.r_dim, self.r_dim))
            self.w_p_o = nn.Parameter(T.zeros(self.r_dim, self.r_dim))
            nn.init.xavier_uniform_(self.w_e_s)
            nn.init.xavier_uniform_(self.w_e_o)
            nn.init.xavier_uniform_(self.w_rp_f)
            nn.init.xavier_uniform_(self.w_rp_i)
            nn.init.xavier_uniform_(self.w_p_s)
            nn.init.xavier_uniform_(self.w_p_o)

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

        if self.t_dim > 0:
            s_emb_s = T.cat((self.e_emb_s(s), self.t_emb_s(s, y, m, d)), 1)
            o_emb_o = T.cat((self.e_emb_o(o), self.t_emb_o(o, y, m, d)), 1)
        else:
            s_emb_s = self.e_emb_s(s)
            o_emb_o = self.e_emb_o(o)
        r_emb_f = self.r_emb_f(r)

        if self.t_dim > 0:
            o_emb_s = T.cat((self.e_emb_s(o), self.t_emb_s(o, y, m, d)), 1)
            s_emb_o = T.cat((self.e_emb_o(s), self.t_emb_o(s, y, m, d)), 1)
        else:
            o_emb_s = self.e_emb_s(o)
            s_emb_o = self.e_emb_o(s)
        r_emb_i = self.r_emb_i(r)

        return s_emb_s, r_emb_f, o_emb_o, o_emb_s, r_emb_i, s_emb_o

    def e_r_emb(self, r, e_t, w_rp):
        e_r = self.p_emb(e_t.view(-1)).view(e_t.size(0), e_t.size(1), self.r_dim).permute(0, 2, 1)
        return e_r @ T.index_select(w_rp, dim=0, index=r.long())

    def forward(self, s, r, o, y, m, d, s_t, s_e, o_t, o_e):
        s_emb_s, r_emb_f, o_emb_o, o_emb_s, r_emb_i, s_emb_o = self.emb(s, r, o, y, m, d)

        a = F.dropout(((s_emb_s * r_emb_f) * o_emb_o +
                       (o_emb_s * r_emb_i) * s_emb_o) / 2.0, p=self.drp, training=self.training).sum(dim=1)

        if self.r_dim > 0:
            s_r_emb_s, o_r_emb_s = self.e_r_emb(r, s_t, self.w_rp_f), self.e_r_emb(r, o_t, self.w_rp_f)
            s_r_emb_o, o_r_emb_o = self.e_r_emb(r, s_t, self.w_rp_i), self.e_r_emb(r, o_t, self.w_rp_i)

            b = ((self.e_emb_s(s).unsqueeze(1) @ self.w_e_s @ o_r_emb_o).squeeze() +
                 (self.e_emb_s(o).unsqueeze(1) @ self.w_e_o @ s_r_emb_o).squeeze()) / 2.0
            c = ((s_r_emb_s.permute(0, 2, 1) @ self.w_e_s.t() @ self.e_emb_o(o).unsqueeze(2)).squeeze() +
                 (o_r_emb_s.permute(0, 2, 1) @ self.w_e_o.t() @ self.e_emb_o(s).unsqueeze(2)).squeeze()) / 2.0
            # d = ((s_r_emb_s.permute(0, 2, 1) @ self.w_p_s @ o_r_emb_o).squeeze() +
            #      (o_r_emb_s.permute(0, 2, 1) @ self.w_p_o @ s_r_emb_o).squeeze()) / 2.0

            return a + b + c + d

        return a

    def l3_reg(self):
        return self.e_emb_s.weight.norm(p=3) ** 3 + self.e_emb_o.weight.norm(p=3) ** 3 + \
            self.m_amp_s.weight.norm(p=3) ** 3 + self.m_amp_o.weight.norm(p=3) ** 3 + \
            self.d_amp_s.weight.norm(p=3) ** 3 + self.d_amp_o.weight.norm(p=3) ** 3 + \
            self.y_amp_s.weight.norm(p=3) ** 3 + self.y_amp_o.weight.norm(p=3) ** 3
