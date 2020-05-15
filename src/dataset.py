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

    def _ix(self):
        ix, s_ix = {}, {}
        for i in range(self.ne):
            ix[i], s_ix[i] = {}, {}
            for j in range(self.nr):
                ix[i][j], s_ix[i][j] = [], {'t': [], 'e': []}
        for s, r, o, y, m, d in self.chk['tr'].astype(np.int):
            t = datetime(year=y, month=m, day=d, tzinfo=pytz.utc).toordinal()
            ix[s][r].append((t, o))
            ix[o][r].append((t, s))  # NOTE: Could have a different list here (-o - 1)
        for k in ix:
            for r in ix[k]:
                s_r = sorted(set(ix[k][r]))
                s_ix[k][r]['t'] = list(map(lambda x: x[0], s_r))
                s_ix[k][r]['e'] = list(map(lambda x: x[1], s_r))
        return s_ix

    def __init__(self, args):
        self.nneg = args.nneg
        self.e2id = {}
        self.r2id = {}
        self.chk = {'tr': self._read(f'data/{args.dataset}/train.txt'),
                    'vd': self._read(f'data/{args.dataset}/valid.txt'),
                    'ts':  self._read(f'data/{args.dataset}/test.txt')}
        self.ne = len(self.e2id)
        self.nr = len(self.r2id)
        self.ix = self._ix()
        self.al = set(map(tuple, np.concatenate(list(self.chk.values())).tolist()))

    def __len__(self):
        return len(self.chk['tr'])

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

    def _rel(self, neg):
        r_s, r_o = [], []
        for s, _, o, y, m, d in neg:
            r_s.append([])
            r_o.append([])
            t = datetime(year=int(y), month=int(m), day=int(d), tzinfo=pytz.utc).toordinal()
            for r in range(self.nr):
                s_ix = bisect.bisect_left(self.ix[s][r]['t'], t) - 1
                o_ix = bisect.bisect_left(self.ix[o][r]['t'], t) - 1
                r_s[-1].append((self.ix[s][r]['t'][s_ix], self.ix[s][r]['e'][s_ix]) if s_ix != -1 else (-1, -1))
                r_o[-1].append((self.ix[o][r]['t'][o_ix], self.ix[o][r]['e'][o_ix]) if o_ix != -1 else (-1, -1))
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
        p = self.chk['tr'][i].reshape(1, -1)
        pn = self._neg(p, self.nneg)
        r_s, r_o = self._rel(pn)
        return self._shred(pn) + self._shred_rel(r_s) + self._shred_rel(r_o)

    def prepare(self, x, md):
        s, r, o, y, m, d = x
        if md == 's':
            x_ts = [(i, r, o, y, m, d) for i in range(self.ne)]
        if md == 'o':
            x_ts = [(s, r, i, y, m, d) for i in range(self.ne)]
        x_ts = np.array([tuple(x)] + list(set(x_ts) - self.al))
        r_s, r_o = self._rel(x_ts)
        return self._shred(x_ts) + self._shred_rel(r_s) + self._shred_rel(r_o)
