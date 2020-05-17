import bisect
import pytz
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as tDataset


class Dataset(tDataset):
    @staticmethod
    def collate_fn(x):
        return [T.cat([_[i] for _ in x]) for i in range(len(x[0]))]

    @staticmethod
    def _id(x2id, x):
        x2id[x] = x2id.get(x, len(x2id))
        return x2id[x]

    def _read(self, fn):
        tups = []
        with open(fn, 'r') as f:
            for line in f.readlines():
                s, r, o, t = line.strip().split('\t')
                t = list(map(float, t.split('-')))
                tups.append([self._id(self.e2id, s), self._id(self.r2id, r), self._id(self.e2id, o), *t])
        return np.array(tups)

    @staticmethod
    def _first(x):
        return x[0]

    @staticmethod
    def _second(x):
        return x[1]

    def _ix(self):
        ix, s_ix = {}, {}
        for i in range(self.ne):
            ix[i], s_ix[i] = {}, {}
            for j in range(self.nr):
                ix[i][j], s_ix[i][j] = [], {'t': [], 'e': []}
        for s, r, o, y, m, d in self.tr.astype(np.int):
            t = datetime(year=y, month=m, day=d, tzinfo=pytz.utc).toordinal()
            ix[s][r].append((t, o))
            ix[o][r].append((t, s))  # NOTE: Could have a different list here (-o - 1)
        for k in ix:
            for r in ix[k]:
                s_r = sorted(set(ix[k][r]))
                s_ix[k][r]['t'] = list(map(self._first, s_r))
                s_ix[k][r]['e'] = list(map(self._second, s_r))
        return s_ix

    def __init__(self, mem, args):
        self.md = 'tr'
        self.mem = mem
        self.nneg = args.nneg
        self.e2id = {}
        self.r2id = {}
        self.tr = self._read(f'data/{args.dataset}/train.txt')
        self.vd = self._read(f'data/{args.dataset}/valid.txt')
        self.ts = self._read(f'data/{args.dataset}/test.txt')
        self.al = set(map(tuple, np.concatenate([self.tr, self.vd, self.ts]).tolist()))
        self.ne = len(self.e2id)
        self.nr = len(self.r2id)
        self.ix = self._ix()

    def train(self):
        self.md = 'tr'

    def valid(self):
        self.md = 'vd'

    def test(self):
        self.md = 'ts'

    def __len__(self):
        return len(getattr(self, self.md))

    def _neg(self, pos, nneg):
        s_neg = np.repeat(pos, nneg + 1, axis=0)
        o_neg = np.repeat(pos, nneg + 1, axis=0)
        s_rnd = np.random.randint(1, self.ne, size=s_neg.shape[0])
        o_rnd = np.random.randint(1, self.ne, size=o_neg.shape[0])
        for i in range(s_neg.shape[0] // (nneg + 1)):
            s_rnd[i * (nneg + 1)] = 0
            o_rnd[i * (nneg + 1)] = 0
        s_neg[:, 0] = (s_neg[:, 0] + s_rnd) % self.ne
        o_neg[:, 2] = (o_neg[:, 2] + o_rnd) % self.ne
        return np.concatenate((s_neg, o_neg), axis=0)

    def _rel_t_map(self, x):
        return datetime(year=int(x[0]), month=int(x[1]), day=int(x[2]), tzinfo=pytz.utc).toordinal()

    def _rel_e_map(self, x):
        mem_k = (x[0], x[1])
        if mem_k in self.mem:
            return self.mem[mem_k]
        r_e = []
        for r in range(self.nr):
            e_ix = bisect.bisect_left(self.ix[x[0]][r]['t'], x[1]) - 1
            r_e.append((self.ix[x[0]][r]['t'][e_ix], self.ix[x[0]][r]['e'][e_ix]) if e_ix != -1 else (-1, -1))
        self.mem[mem_k] = r_e
        return r_e

    def _rel(self, neg):
        t = list(map(self._rel_t_map, neg[:, -3:]))
        r_s = list(map(self._rel_e_map, zip(neg[:, 0], t)))
        r_o = list(map(self._rel_e_map, zip(neg[:, 2], t)))
        return np.array(r_s), np.array(r_o)

    def _shred(self, tup):
        s = T.tensor(tup[:, 0]).long()
        r = T.tensor(tup[:, 1]).long()
        o = T.tensor(tup[:, 2]).long()
        y = T.tensor(tup[:, 3]).float()
        m = T.tensor(tup[:, 4]).float()
        d = T.tensor(tup[:, 5]).float()
        return s, r, o, y, m, d

    def _shred_rel(self, tup):
        t = T.tensor(tup[:, :, 0]).float()
        e = T.tensor(tup[:, :, 1]).long()
        return t, e

    def __getitem__(self, i):
        if self.md == 'tr':
            p = self.tr[i].reshape(1, -1)
            pn = self._neg(p, self.nneg)
            r_s, r_o = self._rel(pn)
            return self._shred(pn) + self._shred_rel(r_s) + self._shred_rel(r_o)
        elif self.md in ['vd', 'ts']:
            x = getattr(self, self.md)[i]
            s, r, o, y, m, d = x
            x_ts_s = np.array([x, ] + [(i, r, o, y, m, d) for i in range(self.ne)])
            x_ts_o = np.array([x, ] + [(s, r, i, y, m, d) for i in range(self.ne)])
            x_ts = np.concatenate([x_ts_s, x_ts_o])
            f_s = np.array([True, ] + [(i, r, o, y, m, d) not in self.al for i in range(self.ne)])
            f_o = np.array([True, ] + [(s, r, i, y, m, d) not in self.al for i in range(self.ne)])
            f = np.concatenate([f_s, f_o])
            r_s, r_o = self._rel(x_ts)
            return self._shred(x_ts) + self._shred_rel(r_s) + self._shred_rel(r_o) + (T.from_numpy(f),)
        else:
            raise ValueError('dataset mode is invalid!')
