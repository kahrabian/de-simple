import torch as T
import torch.nn as nn
import torch.nn.functional as F

from src.models.embedding import PositionalEmbedding


class TFDistMult(nn.Module):
    def __init__(self, ne, nr, args):
        super(TFDistMult, self).__init__()
        self.drp = args.drp

        self.e_emb = nn.Embedding(ne, args.s_dim).to(args.dvc)
        nn.init.xavier_uniform_(self.e_emb.weight)

        self.w_r = nn.Parameter(T.zeros(nr, args.s_dim + args.t_dim, args.s_dim + args.t_dim)).to(args.dvc)
        nn.init.xavier_uniform_(self.w_r)

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

        self.p_emb = PositionalEmbedding(args.r_dim).to(args.dvc)

        self.w_e = nn.Parameter(T.zeros(args.s_dim, args.r_dim)).to(args.dvc)
        nn.init.xavier_uniform_(self.w_e)

        self.w_rp = nn.Parameter(T.zeros(nr, nr, 1)).to(args.dvc)
        nn.init.xavier_uniform_(self.w_rp)

        self.w_p = nn.Parameter(T.zeros(args.r_dim, args.r_dim)).to(args.dvc)
        nn.init.xavier_uniform_(self.w_p)

        self.t_nl = T.sin

    def t_emb(self, e, y, m, d):
        y_emb = self.y_amp(e) * self.t_nl(self.y_frq(e) * y + self.y_phi(e))
        m_emb = self.m_amp(e) * self.t_nl(self.m_frq(e) * m + self.m_phi(e))
        d_emb = self.d_amp(e) * self.t_nl(self.d_frq(e) * d + self.d_phi(e))

        return y_emb + m_emb + d_emb

    def emb(self, s, o, y, m, d):
        y, m, d = y.view(-1, 1), m.view(-1, 1), d.view(-1, 1)

        s_emb = T.cat((self.e_emb(s), self.t_emb(s, y, m, d)), 1)
        o_emb = T.cat((self.e_emb(o), self.t_emb(o, y, m, d)), 1)

        return s_emb, o_emb

    def forward(self, s, r, o, y, m, d, s_t, s_e, o_t, o_e):
        s_emb, o_emb = self.emb(s, o, y, m, d)

        w_r = T.index_select(self.w_r, dim=0, index=r)

        s_p = self.p_emb(s_t.view(-1)).view(s_t.size(0), -1, s_t.size(1))
        o_p = self.p_emb(o_t.view(-1)).view(o_t.size(0), -1, o_t.size(1))

        w_rp = T.index_select(self.w_rp, dim=0, index=r)

        s_p_emb = T.einsum('bij,bjk->bik', (s_p, w_rp))
        o_p_emb = T.einsum('bij,bjk->bik', (o_p, w_rp))

        a = T.einsum('bij,bjk->bik', (T.einsum('bij,bjk->bik', (s_emb.unsqueeze(1), w_r)), o_emb.unsqueeze(2)))
        b = T.einsum('bij,bjk->bik', ((self.e_emb(s) @ self.w_e).unsqueeze(1), s_p_emb))
        c = T.einsum('bij,bjk->bik', ((self.e_emb(o) @ self.w_e).unsqueeze(1), o_p_emb))
        d = T.einsum('bij,bjk->bik', (s_p_emb.permute(0, 2, 1) @ self.w_p, o_p_emb))
        sc = (a + b + c + d).squeeze()

        return sc
