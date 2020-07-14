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
    parser.add_argument('-dataset', type=str, required=True,
                        choices=['gdelt', 'icews14', 'icews05-15', 'GitGraph_TE_0.01', 'GitGraph_TI_0.01'])
    parser.add_argument('-model', type=str, required=True, choices=['DEDistMult', 'DETransE', 'DESimplE', 'TFDistMult'])
    parser.add_argument('-s_dim', type=int, default=20)
    parser.add_argument('-t_dim', type=int, default=64)
    parser.add_argument('-r_dim', type=int, default=16)
    parser.add_argument('-ne', type=int, default=500)
    parser.add_argument('-we', type=int, default=250)
    parser.add_argument('-bs', type=int, default=512)
    parser.add_argument('-tbs', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-lm', type=float, default=0.0)
    parser.add_argument('-nneg', type=int, default=500)
    parser.add_argument('-drp', type=float, default=0.4)
    parser.add_argument('-vd_stp', type=int, default=20)
    parser.add_argument('-mtr', type=str, default='mrr', choices=['h_1', 'h_3', 'h_10', 'mr', 'mrr'])
    parser.add_argument('-md', type=str, default='f', choices=['s', 'o', 'f'])
    parser.add_argument('-te', action='store_true')
    parser.add_argument('-he', action='store_true')
    parser.add_argument('-tr', action='store_true')
    parser.add_argument('-vd', action='store_true')
    parser.add_argument('-ts', action='store_true')
    parser.add_argument('-ch', action='store_true')
    parser.add_argument('-mm', action='store_true')
    parser.add_argument('-w', type=int, default=0)

    return parser.parse_args()


def logger(args):
    log_file = os.path.join(args.pth, 'train.log')
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(log_file, mode='w'), ])


def to(dvc, ts):
    return [t.to(dvc) for t in ts]
