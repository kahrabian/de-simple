import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_emb):
        super(PositionalEmbedding, self).__init__()

        inv_frq = 1 / (10000 ** (torch.arange(0.0, d_emb, 2.0) / d_emb))
        self.register_buffer('inv_frq', inv_frq)

    def forward(self, pos):
        sin = torch.ger(pos, self.inv_frq)
        return torch.cat([sin.sin(), sin.cos()], dim=-1).unsqueeze(1)


class PositionwiseFF(nn.Module):
    def __init__(self, d_in, d_hid, drp, pl_nrm=False):
        super(PositionwiseFF, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(inplace=True),
            nn.Dropout(drp),
            nn.Linear(d_hid, d_in),
            nn.Dropout(drp),
        )
        self.l_nrm = nn.LayerNorm(d_in)
        self.pl_nrm = pl_nrm

    def forward(self, x):
        if self.pl_nrm:
            return self.net(self.l_nrm(x)) + x
        return self.l_nrm(x + self.net(x))


class MultiHeadAttn(nn.Module):
    def __init__(self, n_h, d_in, d_h, drp, drp_att=0, pl_nrm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_h = n_h
        self.d_h = d_h

        self.qkv_net = nn.Linear(d_in, 3 * self.n_h * self.d_h, bias=False)

        self.drp = nn.Dropout(drp)
        self.drp_att = nn.Dropout(drp_att)
        self.o_net = nn.Linear(self.n_h * self.d_h, d_in, bias=False)

        self.l_nrm = nn.LayerNorm(d_in)

        self.sc = 1 / (self.d_h ** 0.5)

        self.pl_nrm = pl_nrm

        self.r_net = nn.Linear(d_in, self.n_h * self.d_h, bias=False)

    def _rel_shift(self, x):
        pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1), device=x.device, dtype=x.dtype)
        return torch.cat([pad, x], dim=3).view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))[:, :, 1:].view_as(x)

    def forward(self, w, r, r_w_b, r_r_b, att_msk=None, m=None):
        q_l, r_l = w.size(0), r.size(0)

        if m is not None:
            _m = torch.cat([m, w], 0)
            if self.pl_nrm:
                w_qkv = self.qkv_net(self.l_nrm(_m))
            else:
                w_qkv = self.qkv_net(_m)

            w_q, w_k, w_v = torch.chunk(w_qkv, 3, dim=-1)
            w_q = w_q[-q_l:]
        else:
            if self.pl_nrm:
                w_qkv = self.qkv_net(self.l_nrm(w))
            else:
                w_qkv = self.qkv_net(w)

            w_q, w_k, w_v = torch.chunk(w_qkv, 3, dim=-1)

        r_k = self.r_net(r)

        k_l = w_k.size(0)

        w_q = w_q.view(q_l, -1, self.n_h, self.d_h)  # qlen x bsz x n_head x d_head
        w_k = w_k.view(k_l, -1, self.n_h, self.d_h)  # klen x bsz x n_head x d_head
        w_v = w_v.view(k_l, -1, self.n_h, self.d_h)  # klen x bsz x n_head x d_head

        r_k = r_k.view(r_l, -1, self.n_h, self.d_h)  # qlen x bsz x n_head x d_head

        # compute attention score
        rw_q = w_q + r_w_b                                  # qlen x bsz x n_head x d_head
        ac = torch.einsum('ibnd,jbnd->bnij', (rw_q, w_k))   # bsz x n_head x qlen x klen

        rr_q = w_q + r_r_b                                  # qlen x bsz x n_head x d_head
        bd = torch.einsum('ibnd,jbnd->bnij', (rr_q, r_k))   # bsz x n_head x qlen x klen
        bd = self._rel_shift(bd)

        # [bsz x n_head x qlen x klen]
        att_sc = ac + bd
        att_sc.mul_(self.sc)

        # compute attention probability
        if att_msk is not None and att_msk.any().item():
            if att_msk.dim() == 2:
                att_sc.masked_fill_(att_msk[None, None, :, :], -float('inf'))
            elif att_msk.dim() == 3:
                att_sc.masked_fill_(att_msk[:, None, :, :], -float('inf'))

        # [bsz x n_head x qlen x klen]
        att_p = F.softmax(att_sc, dim=3)
        att_p = self.drp_att(att_p)

        # compute attention vector
        att_v = torch.einsum('bnij,jbnd->ibnd', (att_p, w_v))

        # [qlen x bsz x n_head x d_head]
        att_v = att_v.contiguous().view(att_v.size(0), att_v.size(1), self.n_h * self.d_h)

        # linear projection
        att_o = self.o_net(att_v)
        att_o = self.drp(att_o)

        if self.pl_nrm:
            return w + att_o
        return self.l_nrm(w + att_o)


class DecoderLayer(nn.Module):
    def __init__(self, n_h, d_in, d_h, d_hid, drp, pl_nrm=False, **kwargs):
        super(DecoderLayer, self).__init__()

        self.att = MultiHeadAttn(n_h, d_in, d_h, drp, pl_nrm=pl_nrm, **kwargs)
        self.pff = PositionwiseFF(d_in, d_hid, drp, pl_nrm)

    def forward(self, x, r, r_w_b, r_r_b, att_msk=None, m=None):
        out = self.att(x, r, r_w_b, r_r_b, att_msk=att_msk, m=m)
        return self.pff(out)


class MemTransformerXL(nn.Module):
    def __init__(self, n_l, n_h, d_in, d_h, d_hid, drp, drp_att, pl_nrm=False, m_l=None, clm_l=-1):
        super(MemTransformerXL, self).__init__()

        self.ls = nn.ModuleList()
        for i in range(n_l):
            self.ls.append(DecoderLayer(n_h, d_in, d_h, d_hid, drp, drp_att=drp_att, pl_nrm=pl_nrm))

        self.drp = nn.Dropout(drp)

        self.p_emb = PositionalEmbedding(d_in)
        self.r_w_b = nn.Parameter(torch.Tensor(n_h, d_h))
        self.r_r_b = nn.Parameter(torch.Tensor(n_h, d_h))

        self.m_l = m_l
        self.clm_l = clm_l

    def _m_init(self):
        if self.m_l > 0:
            p = next(self.parameters())
            m = [torch.empty(0, dtype=p.dtype, device=p.device) for i in range(len(self.ls) + 1)]
            return m
        return None

    def _m_update(self, h, m, q_l, m_l):
        if m is None:
            return None

        assert len(h) == len(m), 'len(h) != len(m)'

        with torch.no_grad():
            e_ix = m_l + q_l
            s_ix = max(0, e_ix - self.m_l)
            _m = [torch.cat([m[i], h[i]], dim=0)[s_ix:e_ix].detach() for i in range(len(h))]
        return _m

    def forward(self, sq, p, m):
        if m is None:
            m = self._m_init()

        q_l = sq.size(0)
        m_l = m[0].size(0) if m is not None else 0
        k_l = m_l + q_l

        att_msk = torch.triu(sq.new_ones(q_l, k_l), 1 + m_l).bool()

        p = torch.arange(k_l - 1, -1, -1.0, device=sq.device, dtype=sq.dtype)
        if self.clm_l > 0:
            p.clamp_(max=self.clm_l)
        # p_emb = self.drp(self.p_emb(p.view(-1)).view(q_l, sq.size(1), -1))
        # print(p_emb.shape)
        p_emb = self.drp(self.p_emb(p))
        # print(p_emb.shape)
        # exit()

        h = self.drp(sq)
        hs = [h, ]
        for i, layer in enumerate(self.ls):
            m_i = None if m is None else m[i]
            h = layer(h, p_emb, self.r_w_b, self.r_r_b, att_msk=att_msk, m=m_i)
            hs.append(h)

        h = self.drp(h)
        _m = self._m_update(hs, m, q_l, m_l)

        return h, _m
