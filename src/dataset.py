import bisect
import itertools
import pytz
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as tDataset


class Dataset(tDataset):
    _t_regx = re.compile(r'^/(\w+)/.*$')

    @staticmethod
    def collate_fn(x):
        return [T.cat([_[i] for _ in x]) for i in range(len(x[0]))]

    @staticmethod
    def _id(x2id, x):
        x2id[x] = x2id.get(x, len(x2id))
        return x2id[x]

    def _t_id(self, x, x_id):
        t = re.findall(self._t_regx, x)[0]
        self.e2t[x_id] = t
        if t not in self.t2id:
            self.t2id[t] = set()
        self.t2id[t].add(x_id)

    def _read(self, fn):
        tups = []
        with open(fn, 'r') as f:
            for line in f.readlines():
                s, r, o, t = line.strip().split('\t')
                s_id, r_id, o_id = self._id(self.e2id, s), self._id(self.r2id, r), self._id(self.e2id, o)
                t = list(map(float, t.split('-')))
                tups.append([s_id, r_id, o_id, *t])
                if self.te:
                    self._t_id(s, s_id)
                    self._t_id(o, o_id)
        return np.array(tups).astype(np.int)

    @staticmethod
    def _first(x):
        return x[0]

    @staticmethod
    def _second(x):
        return x[1]

    def _h_ix(self, ds):
        tr = pd.read_csv(f'data/{ds}/train.txt', sep='\t', names=['s', 'r', 'o', 't'])

        r_e = tr[tr['o'].str.startswith('/repo/')][['o', 's']]
        r_e = r_e.rename(columns={'o': 'repo', 's': 'entity'})

        e_u = tr[tr['s'].str.startswith('/user/')].groupby('o')['s'].apply(list)
        e_u = e_u.reset_index(name='users').rename(columns={'o': 'entity'})

        r_u = r_e.merge(e_u, on='entity', how='left')[['repo', 'users']]
        r_u['users'] = r_u.users.apply(lambda x: x if type(x) == list else [])
        r_u = r_u.groupby('repo')['users'].apply(lambda x: list(itertools.chain.from_iterable(x)))
        r_u = r_u.reset_index(name='users')

        vd = pd.read_csv(f'data/{ds}/valid.txt', sep='\t', names=['s', 'r', 'o', 't'])
        ts = pd.read_csv(f'data/{ds}/test.txt', sep='\t', names=['s', 'r', 'o', 't'])
        al = pd.concat([tr, vd, ts])

        i_r = al[al['o'].str.startswith('/repo/') & al['s'].str.startswith('/issue/')]
        i_r = i_r.groupby('s')['o'].apply(lambda x: list(x)[0]).reset_index(name='repo')
        i_r = i_r.rename(columns={'s': 'issue'})

        p_r = al[al['o'].str.startswith('/repo/') & al['s'].str.startswith('/pr/')]
        p_r = p_r.groupby('s')['o'].apply(lambda x: list(x)[0]).reset_index(name='repo')
        p_r = p_r.rename(columns={'s': 'pr'})

        i_u = i_r.merge(r_u, on='repo', how='left')[['issue', 'users']]
        i_u['issue'] = i_u.issue.apply(lambda x: int(self.e2id[x]))
        i_u['users'] = i_u.users.fillna('').apply(lambda x: [int(self.e2id[y]) for y in x])
        i_u_ix = i_u.set_index('issue').to_dict()['users']

        p_u = p_r.merge(r_u, on='repo', how='left')[['pr', 'users']]
        p_u['pr'] = p_u.pr.apply(lambda x: int(self.e2id[x]))
        p_u['users'] = p_u.users.fillna('').apply(lambda x: [int(self.e2id[y]) for y in x])
        p_u_ix = p_u.set_index('pr').to_dict()['users']

        return {**i_u_ix, **p_u_ix}

    def _ix(self):
        ix, s_ix = {}, {}
        for i in range(self.ne):
            ix[i], s_ix[i] = {}, {}
            for j in range(self.nr):
                ix[i][j], s_ix[i][j] = [], {'t': [], 'e': []}
        for s, r, o, y, m, d in self.tr.astype(np.int):
            t = self._rel_t_map((y, m, d))
            ix[s][r].append((t, o))
            ix[o][r].append((t, s))  # NOTE: Could have a different list here (-o - 1)
        for k in ix:
            for r in ix[k]:
                s_r = sorted(set(ix[k][r]))
                s_ix[k][r]['t'] = list(map(self._first, s_r))
                s_ix[k][r]['e'] = list(map(self._second, s_r))
        return s_ix

    def __init__(self, mem, args):
        self.l_md = 'tr'
        self.e_md = args.md
        self.mem = mem
        self.nneg = args.nneg
        self.e2id = {}
        self.r2id = {}
        self.te = args.te
        if self.te:
            self.e2t = {}
            self.t2id = {}
        self.tr = self._read(f'data/{args.dataset}/train.txt')
        self.vd = self._read(f'data/{args.dataset}/valid.txt')
        self.ts = self._read(f'data/{args.dataset}/test.txt')
        self.al = set(map(tuple, np.concatenate([self.tr, self.vd, self.ts]).tolist()))
        self.he = args.he
        if self.he:
            self.h_ix = self._h_ix(args.dataset)
        self.ne = len(self.e2id)
        self.nr = len(self.r2id)
        self.ix = self._ix()
        self.t_pr = min(map(lambda x: self._rel_t_map(x[-3:]), self.tr)) - \
            max(map(lambda x: self._rel_t_map(x[-3:]), self.tr))

    def train(self):
        self.l_md = 'tr'

    def valid(self):
        self.l_md = 'vd'

    def test(self):
        self.l_md = 'ts'

    def __len__(self):
        return len(getattr(self, self.l_md))

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
            i = bisect.bisect_left(self.ix[x[0]][r]['t'], x[1]) - 1
            r_e.append((x[1] - self.ix[x[0]][r]['t'][i], self.ix[x[0]][r]['e'][i]) if i != -1 else (self.t_pr, -1))
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

    def _filter(self, x, y, e, i):
        return (x not in self.al) and (not self.te or i in self.t2id[self.e2t[y]]) and \
            (not self.he or y not in self.h_ix.get(e, []) or i in self.h_ix.get(e, []))

    def __getitem__(self, i):
        if self.l_md == 'tr':
            p = self.tr[i].reshape(1, -1)
            pn = self._neg(p, self.nneg)
            r_s, r_o = self._rel(pn)
            return self._shred(pn) + self._shred_rel(r_s) + self._shred_rel(r_o)
        elif self.l_md in ['vd', 'ts']:
            x = getattr(self, self.l_md)[i]
            s, r, o, y, m, d = x
            x_ts, f = [], []
            if self.e_md in ['f', 's']:
                x_ts.append(np.array([x, ] + [(i, r, o, y, m, d) for i in range(self.ne)]))
                f.append(np.array([False, ] + [self._filter((i, r, o, y, m, d), s, o, i) for i in range(self.ne)]))
            if self.e_md in ['f', 'o']:
                x_ts.append(np.array([x, ] + [(s, r, i, y, m, d) for i in range(self.ne)]))
                f.append(np.array([False, ] + [self._filter((s, r, i, y, m, d), o, s, i) for i in range(self.ne)]))
            x_ts = np.concatenate(x_ts)
            f = np.concatenate(f)
            r_s, r_o = self._rel(x_ts)
            return self._shred(x_ts) + self._shred_rel(r_s) + self._shred_rel(r_o) + (T.from_numpy(f),)
        else:
            raise ValueError('dataset mode is invalid!')
