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
        ix = defaultdict(lambda: defaultdict(list))
        for s, r, o, y, m, d in self._ds['tr'].astype(np.int):
            t = datetime(year=y, month=m, day=d, tzinfo=pytz.utc).toordinal()
            ix[s][r].append((t, o))
            ix[o][r].append((t, s))  # NOTE: Could have a different list here (-o - 1)
        s_ix = defaultdict(lambda: defaultdict(lambda: {'t': [], 'e': []}))
        for k in ix:
            for r in ix[k]:
                s_r = sorted(set(ix[k][r]))
                s_ix[k][r]['t'] = list(map(lambda x: x[0], s_r))
                s_ix[k][r]['e'] = list(map(lambda x: x[1], s_r))
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
            r_s.append([])
            r_o.append([])
            t = datetime(year=int(y), month=int(m), day=int(d), tzinfo=pytz.utc).toordinal()
            for r in range(self.nr):
                s_ix = bisect.bisect_left(self.ix[s][r]['t'], t) - 1
                o_ix = bisect.bisect_left(self.ix[o][r]['t'], t) - 1
                r_s[-1].append((self.ix[s][r]['t'][s_ix], self.ix[s][r]['e'][s_ix]) if s_ix != -1 else (-1, -1))
                r_o[-1].append((self.ix[o][r]['t'][o_ix], self.ix[o][r]['e'][o_ix]) if o_ix != -1 else (-1, -1))
        return np.array(r_s), np.array(r_o)

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
