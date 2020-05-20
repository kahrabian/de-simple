import torch as T
import torch.nn as nn
import torch.nn.functional as F

from src.models.embedding import PositionalEmbedding


class DEDistMult(nn.Module):
    def __init__(self, ne, nr, args):
        super(DEDistMult, self).__init__()
        self.drp = args.drp
        self.rel = args.r_dim > 0

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

        if self.rel:
            self.p_emb = PositionalEmbedding(args.r_dim)

            self.w_p = nn.Linear(nr, 1, bias=False)
            nn.init.xavier_uniform_(self.w_p.weight)

            self.w_r = nn.Parameter(T.zeros(nr, args.s_dim + args.t_dim, args.r_dim))
            nn.init.xavier_uniform_(self.w_r)

            self.pr_emb = nn.Embedding(nr, args.r_dim).to(args.dvc)
            nn.init.xavier_uniform_(self.pr_emb.weight)

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

    def forward(self, s, r, o, y, m, d, s_t, s_e, o_t, o_e):
        s_emb, r_emb, o_emb = self.emb(s, r, o, y, m, d)

        if self.rel:
            s_p_emb = self.w_p(self.p_emb(s_t.view(-1)).view(s_t.size(0), -1, s_t.size(1))).squeeze()
            o_p_emb = self.w_p(self.p_emb(o_t.view(-1)).view(o_t.size(0), -1, o_t.size(1))).squeeze()

            w_r = T.index_select(self.w_r, dim=0, index=r)

            sc = T.cat([(s_emb * r_emb) * o_emb,
                        T.einsum('be,bpe->bp', (s_p_emb, w_r)) * o_emb,
                        s_emb * T.einsum('be,bpe->bp', (o_p_emb, w_r)),
                        s_p_emb * self.pr_emb(r) * o_p_emb], dim=1)
        else:
            sc = (s_emb * r_emb) * o_emb
        sc = F.dropout(sc, p=self.drp, training=self.training)
        sc = T.sum(sc, dim=1)

        return sc
