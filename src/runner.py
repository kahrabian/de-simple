import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


import src.utils as ut
from src import models


class Runner(object):
    def __init__(self, ds, args):
        self.args = args
        self.ds = ds
        self.mdl = nn.DataParallel(getattr(models, args.model)(ds.ne, ds.nr, args))
        self.mtr = -1
        self.tb_sw = SummaryWriter()

    def log_tensorboard(self, chk, mtrs, e):
        for mtr, val in dict(mtrs).items():
            self.tb_sw.add_scalar(f'{chk}/{mtr}', val, e)

    def train(self):
        opt = torch.optim.Adam(self.mdl.parameters(), lr=self.args.lr)
        ls_f = nn.CrossEntropyLoss()

        for e in range(1, self.args.ne + 1):
            self.ds.reset()
            self.mdl.train()

            stp = 0
            tot_ls = 0.0
            with tqdm(total=int(np.ceil(len(self.ds) / self.args.bs)), desc=f'epoch {e}/{self.args.ne}') as pb:
                while self.ds.loc >= 0:
                    opt.zero_grad()
                    s, r, o, y, m, d = self.ds.next(self.args.bs, self.args.nneg, self.args.dvc)
                    sc = self.mdl(s, r, o, y, m, d).view(-1, self.args.nneg + 1)
                    ls = ls_f(sc, torch.zeros(sc.size(0)).long().to(self.args.dvc))
                    ls.backward()
                    opt.step()
                    tot_ls += ls.item()

                    stp += 1
                    self.tb_sw.add_scalar(f'epoch/{e}/loss', ls.item(), stp)
                    pb.set_postfix(loss=f'{tot_ls / stp:.6f}')
                    pb.update()

            self.log_tensorboard('train', {'loss': tot_ls / stp}, e)
            logging.info(f'epoch {e}/{self.args.ne}: loss={tot_ls / stp:.6f}')

            if self.args.vd and (e % self.args.vd_stp == 0 or e == self.args.ne):
                mtrs = self.test('vd')
                self.log_tensorboard('valid', mtrs, e)
                logging.info(f'epoch {e}/{self.args.ne} Validation: {mtrs}')
                if getattr(mtrs, self.args.mtr) > self.mtr:
                    self.save(self.mdl, opt)

    def _prepare(self, x, md):
        s, r, o, y, m, d = x
        if md == 's':
            x_vd = [(i, r, o, y, m, d) for i in range(self.ds.ne)]
        if md == 'o':
            x_vd = [(s, r, i, y, m, d) for i in range(self.ds.ne)]
        return ut.shred(np.array([tuple(x)] + list(set(x_vd) - self.ds.al)), self.args.dvc)

    def test(self, chk):
        self.mdl.eval()
        mtrs = ut.Metric()
        with tqdm(total=len(self.ds._ds[chk]), desc='valid' if chk == 'vd' else 'test') as pb:
            for x in self.ds._ds[chk]:
                for md in ['s', 'o']:
                    s, r, o, y, m, d = self._prepare(x, md)
                    sc = self.mdl(s, r, o, y, m, d).detach().cpu().numpy()
                    rk = (sc > sc[0]).sum() + 1
                    mtrs.update(rk)
                pb.set_postfix(**dict(mtrs))
                pb.update()
        return mtrs

    def save(self, mdl, opt):
        with open(os.path.join(self.args.pth, 'cfg.json'), 'w') as f:
            json.dump(vars(self.args), f, indent=4, sort_keys=True)
        torch.save({self.args.mtr: self.mtr,
                    'mdl': mdl.state_dict(),
                    'opt': opt.state_dict()}, os.path.join(self.args.pth, f'chk_{self.args.mtr}.dat'))
