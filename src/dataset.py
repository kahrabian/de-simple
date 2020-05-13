import bisect
import pytz
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch as T
from torch.nn.utils.rnn import pad_sequence

from src import utils as ut


class Dataset(object):
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
                tups.append([self._id(self._e2id, s), self._id(self._r2id, r), self._id(self._e2id, o), *t])
        return np.array(tups)

    def _ix(self):
        ix = defaultdict(list)
        for s, r, o, y, m, d in self._ds['tr'].astype(np.int):
            t = datetime(year=y, month=m, day=d, tzinfo=pytz.utc).toordinal()
            ix[s].append((t, r, o))
            ix[-o - 1].append((t, r, o))
        s_ix = defaultdict(lambda: np.array([[], ] * 3))
        for k, v in ix.items():
            s_ix[k] = np.array(sorted(v)).T
        return s_ix

    def __init__(self, ds):
        self.loc = 0
        self._e2id = {}
        self._r2id = {}
        self._ds = {'tr': self._read(f'data/{ds}/train.txt'),
                    'vd': self._read(f'data/{ds}/valid.txt'),
                    'ts':  self._read(f'data/{ds}/test.txt')}
        self.ix = self._ix()
        self.ne = len(self._e2id)
        self.nr = len(self._r2id)
        self.al = set(map(tuple, np.concatenate(list(self._ds.values())).tolist()))

    def __len__(self):
        return len(self._ds['tr'])

    def reset(self):
        self.loc = 0

    def _pos(self, bs):
        if self.loc + bs <= len(self._ds['tr']):
            pos = self._ds['tr'][self.loc:self.loc + bs]
            self.loc += bs
        else:
            pos = self._ds['tr'][self.loc:]
            self.loc = -1
        return pos

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

    def _rel(self, neg, dvc):
        r_s, r_o = [], []
        for s, _, o, y, m, d in neg:
            t = datetime(year=int(y), month=int(m), day=int(d), tzinfo=pytz.utc).toordinal()
            s_ix = bisect.bisect_left(self.ix[s][0], t) - 1
            o_ix = bisect.bisect_left(self.ix[-o - 1][0], t) - 1
            r_s.append(T.from_numpy((self.ix[s][:, :s_ix] if s_ix != -1 else np.array([[], ] * 3)).T))
            r_o.append(T.from_numpy((self.ix[-o - 1][:, :o_ix] if o_ix != -1 else np.array([[], ] * 3)).T))
        r_s = pad_sequence(r_s).permute(1, 2, 0).numpy()
        r_o = pad_sequence(r_o).permute(1, 2, 0).numpy()
        return r_s, r_o

    def next(self, bs, nneg, dvc):
        p = self._pos(bs)
        pn = self._neg(p, nneg)
        r_s, r_o = self._rel(pn, dvc)
        return ut.shred(pn, dvc) + ut.shred_rel(r_s, dvc) + ut.shred_rel(r_o, dvc)

    def prepare(self, x, md, dvc):
        s, r, o, y, m, d = x
        if md == 's':
            x_ts = [(i, r, o, y, m, d) for i in range(self.ne)]
        if md == 'o':
            x_ts = [(s, r, i, y, m, d) for i in range(self.ne)]
        x_ts = np.array([tuple(x)] + list(set(x_ts) - self.al))
        r_s, r_o = self._rel(x_ts, dvc)
        return ut.shred(x_ts, dvc) + ut.shred_rel(r_s, dvc) + ut.shred_rel(r_o, dvc)
