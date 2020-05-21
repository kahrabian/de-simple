import json
import logging
import os
import pickle
import shutil

import numpy as np
import torch as T
import torch.nn as nn
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


import src.utils as ut
from src import models
from src.dataset import Dataset


class Runner(object):
    def __init__(self, args):
        self.args = args
        self.mem = self.load_mem()
        self.ds = Dataset(self.mem, self.args)
        self.mdl = nn.DataParallel(getattr(models, self.args.model)(self.ds.ne, self.ds.nr, self.args))
        self.mtrs = ut.Metric()
        self.tb_sw = SummaryWriter()

    def log_tensorboard(self, chk, mtrs, e):
        for mtr, val in dict(mtrs).items():
            self.tb_sw.add_scalar(f'{chk}/{mtr}', val, e)

    def train(self):
        opt = T.optim.Adam(self.mdl.parameters(), lr=self.args.lr)
        lr_sc = StepLR(opt, self.args.we)
        ls_f = nn.CrossEntropyLoss()

        dl = DataLoader(self.ds, batch_size=self.args.bs, num_workers=self.args.w, pin_memory=True,
                        collate_fn=Dataset.collate_fn)
        for e in range(1, self.args.ne + 1):
            self.mdl.train()
            self.ds.train()

            avg_ls = 0.0
            with tqdm(total=len(dl), desc=f'epoch {e}/{self.args.ne}') as pb:
                for i, x in enumerate(dl, 1):
                    if not self.args.ch:
                        opt.zero_grad()
                        s, r, o, y, m, d, s_t, s_e, o_t, o_e = ut.to(self.args.dvc, x)
                        sc = self.mdl(s, r, o, y, m, d, s_t, s_e, o_t, o_e).view(-1, self.args.nneg + 1)
                        ls = ls_f(sc, T.zeros(sc.size(0)).long().to(self.args.dvc))
                        ls.backward()
                        opt.step()
                        avg_ls += ls.item()
                        self.tb_sw.add_scalar(f'epoch/loss/{e}', ls.item(), i)
                        self.tb_sw.add_scalar('train/loss', ls.item(), (e - 1) * len(dl) + i)
                        pb.set_postfix(loss=f'{avg_ls / i:.6f}')
                    pb.update()

            lr_sc.step()

            self.save_mem()
            self.log_tensorboard('train', {'loss': avg_ls / len(dl)}, e)
            logging.info(f'epoch {e}/{self.args.ne}: loss={avg_ls / len(dl):.6f}')

            if self.args.vd and (e % self.args.vd_stp == 0 or e == self.args.ne):
                self.valid(e, opt)

        if not self.args.ch:
            self.tb_sw.add_hparams(vars(self.args), dict(self.mtrs))

    def eval(self, desc):
        self.mdl.eval()
        mtrs = ut.Metric()
        dl = DataLoader(self.ds, batch_size=self.args.tbs, num_workers=self.args.w, pin_memory=True,
                        collate_fn=Dataset.collate_fn)
        with tqdm(total=len(dl), desc=desc) as pb:
            for i, x in enumerate(dl, 1):
                if not self.args.ch:
                    s, r, o, y, m, d, s_t, s_e, o_t, o_e, f = ut.to(self.args.dvc, x)
                    sc = self.mdl(s, r, o, y, m, d, s_t, s_e, o_t, o_e).view(-1, self.ds.ne + 1).detach().cpu().numpy()
                    for sc_i, f_i in zip(sc, f.view(-1, self.ds.ne + 1).detach().cpu().numpy()):
                        mtrs.update((sc_i[f_i] > sc_i[0]).sum() + 1)
                    pb.set_postfix(**dict(mtrs))
                pb.update()
        return mtrs

    def test(self):
        with T.no_grad():
            if not self.args.ch:
                self.load()
            self.ds.test()
            mtrs = self.eval('test')
            self.save_mem()
            if not self.args.ch:
                self.log_tensorboard('test', mtrs, 0)
                logging.info(f'Test: {mtrs}')

    def valid(self, e, opt):
        with T.no_grad():
            self.ds.valid()
            mtrs = self.eval('valid')
            self.save_mem()
            if not self.args.ch:
                self.log_tensorboard('valid', mtrs, e)
                logging.info(f'epoch {e}/{self.args.ne} Validation: {mtrs}')
                if self.mtrs.cnt == 0 or getattr(mtrs, self.args.mtr) > getattr(self.mtrs, self.args.mtr):
                    self.mtrs = mtrs
                    self.save(opt)

    def load(self, opt=None):
        p = os.path.join(self.args.pth, f'chk_{self.args.mtr}.dat')
        if not os.path.exists(p):
            return
        chk = T.load(p)
        self.mdl.load_state_dict(chk['mdl'])
        if opt is not None:
            opt.load_state_dict(chk['opt'])

    def save(self, opt):
        with open(os.path.join(self.args.pth, 'cfg.json'), 'w') as f:
            json.dump(vars(self.args), f, indent=4, sort_keys=True)
        T.save({**dict(self.mtrs),
                'mdl': self.mdl.state_dict(),
                'opt': opt.state_dict()}, os.path.join(self.args.pth, f'chk_{self.args.mtr}.dat'))

    def load_mem(self):
        if not os.path.exists(self.args.mem_pth):
            return {}
        with open(self.args.mem_pth, 'rb') as f:
            return pickle.load(f)

    def save_mem(self):
        with open(f'{self.args.mem_pth}_{self.args.id}_tmp', 'wb') as f:
            pickle.dump(self.mem, f, protocol=pickle.HIGHEST_PROTOCOL)
        shutil.copy(f'{self.args.mem_pth}_{self.args.id}_tmp', f'{self.args.mem_pth}_{self.args.id}')
