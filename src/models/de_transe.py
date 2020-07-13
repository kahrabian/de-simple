import torch as T
import torch.nn as nn
import torch.nn.functional as F

from src.models.embedding import PositionalEmbedding


class DETransE(nn.Module):
    def __init__(self, ne, nr, args):
        super(DETransE, self).__init__()
        self.drp = args.drp
        self.r_dim = args.r_dim

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

        if self.r_dim > 0:
            self.p_emb = PositionalEmbedding(self.r_dim).to(args.dvc)

            self.w_e = nn.Parameter(T.zeros(args.s_dim + args.t_dim, self.r_dim)).to(args.dvc)
            self.w_rp = nn.Parameter(T.zeros(nr, nr, 1)).to(args.dvc)
            nn.init.xavier_uniform_(self.w_e)
            nn.init.xavier_uniform_(self.w_rp)

        self.t_nl = T.sin

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

    def e_r_emb(self, r, e_t):
        e_r = self.p_emb(e_t.view(-1)).view(e_t.size(0), e_t.size(1), self.r_dim).permute(0, 2, 1)
        return e_r @ T.index_select(self.w_rp, dim=0, index=r.long())

    def forward(self, s, r, o, y, m, d, s_t, s_e, o_t, o_e):
        s_emb, r_emb, o_emb = self.emb(s, r, o, y, m, d)

        a = F.dropout((s_emb + r_emb) - o_emb, p=self.drp, training=self.training).norm(dim=1)

        if self.r_dim > 0:
            s_r_emb, o_r_emb = self.e_r_emb(r, s_t), self.e_r_emb(r, o_t)

            b = F.dropout(self.e_emb(s) @ self.w_e - o_r_emb.squeeze(), p=self.drp, training=self.training).norm(dim=1)
            c = F.dropout(s_r_emb.squeeze() - self.e_emb(o) @ self.w_e, p=self.drp, training=self.training).norm(dim=1)

            return (-1) * (a + b + c)

        return (-1) * a

    def l3_reg(self):
        return self.e_emb.weight.norm(p=3) ** 3 + \
            self.m_amp.weight.norm(p=3) ** 3 + \
            self.d_amp.weight.norm(p=3) ** 3 + \
            self.y_amp.weight.norm(p=3) ** 3
