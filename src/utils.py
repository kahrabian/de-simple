import argparse
import logging
import os

import torch as T


class Metric(object):
    def __init__(self):
        self.cnt = 0
        self.h_1 = 0
        self.h_3 = 0
        self.h_10 = 0
        self.mr = 0
        self.mrr = 0

    def _normalize(self):
        return self.h_1 / self.cnt, self.h_3 / self.cnt, self.h_10 / self.cnt, self.mr / self.cnt, self.mrr / self.cnt

    def __str__(self):
        h_1, h_3, h_10, mr, mrr = self._normalize()
        return f'H@1: {h_1}\tH@3: {h_3}\tH@10: {h_10}\tMR: {mr}\tMRR: {mrr}'

    def __iter__(self):
        h_1, h_3, h_10, mr, mrr = self._normalize()
        yield 'metric/H1', h_1
        yield 'metric/H3', h_3
        yield 'metric/H10', h_10
        yield 'metric/MR', mr
        yield 'metric/MRR', mrr

    def update(self, r):
        self.cnt += 1
        if r <= 1:
            self.h_1 += 1
        if r <= 3:
            self.h_3 += 1
        if r <= 10:
            self.h_10 += 1
        self.mr += r
        self.mrr += 1.0 / r


def args():
    parser = argparse.ArgumentParser(description='Temporal KG Completion')

    parser.add_argument('-id', type=str, required=True)
    parser.add_argument('-dataset', type=str, default='icews14', choices=['gdelt', 'icews14', 'icews05-15'])
    parser.add_argument('-model', type=str, default='DEDistMult', choices=['DEDistMult', 'DETransE', 'DESimplE'])
    parser.add_argument('-s_dim', type=int, default=36)
    parser.add_argument('-t_dim', type=int, default=64)

    parser.add_argument('-nl', type=int, default=4)
    parser.add_argument('-nh', type=int, default=8)
    parser.add_argument('-ql', type=int, default=32)
    parser.add_argument('-ml', type=int, default=64)

    parser.add_argument('-ne', type=int, default=500)
    parser.add_argument('-bs', type=int, default=512)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-nneg', type=int, default=500)
    parser.add_argument('-drp', type=float, default=0.4)
    parser.add_argument('-vd_stp', type=int, default=20)
    parser.add_argument('-mtr', type=str, default='mrr', choices=['h_1', 'h_3', 'h_10', 'mr', 'mrr'])
    parser.add_argument('-tr', action='store_true')
    parser.add_argument('-vd', action='store_true')
    parser.add_argument('-ts', action='store_true')

    return parser.parse_args()


def logger(args):
    log_file = os.path.join(args.pth, 'train.log')
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(log_file, mode='w'), ])


def shred(tup):
    s = T.tensor(tup[:, 0]).long()
    r = T.tensor(tup[:, 1]).long()
    o = T.tensor(tup[:, 2]).long()
    y = T.tensor(tup[:, 3]).float()
    m = T.tensor(tup[:, 4]).float()
    d = T.tensor(tup[:, 5]).float()
    return s, r, o, y, m, d


def shred_rel(tup):
    t = T.tensor(tup[:, 0]).float()
    r = T.tensor(tup[:, 1]).long()
    e = T.tensor(tup[:, 2]).long()
    return t, r, e
