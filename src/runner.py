import json
import logging
import os
import pickle
import shutil

import numpy as np
import torch as T
import torch.nn as nn
import torch.multiprocessing as mp
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
        self.dl = DataLoader(self.ds, batch_size=self.args.bs, shuffle=True, num_workers=os.cpu_count(),
                             pin_memory=True, collate_fn=Dataset.collate_fn)
        self.mdl = nn.DataParallel(getattr(models, self.args.model)(self.ds.ne, self.ds.nr, self.args))
        self.mtrs = ut.Metric()
        self.tb_sw = SummaryWriter()

    def log_tensorboard(self, chk, mtrs, e):
        for mtr, val in dict(mtrs).items():
            self.tb_sw.add_scalar(f'{chk}/{mtr}', val, e)

    def train(self):
        opt = T.optim.Adam(self.mdl.parameters(), lr=self.args.lr)
        ls_f = nn.CrossEntropyLoss()

        for e in range(1, self.args.ne + 1):
            self.mdl.train()

            tot_ls = 0.0
            with tqdm(total=len(self.dl), desc=f'epoch {e}/{self.args.ne}') as pb:
                for i, x in enumerate(self.dl, 1):
                    opt.zero_grad()
                    s, r, o, y, m, d, s_t, s_e, o_t, o_e = ut.to(self.args.dvc, x)
                    sc = self.mdl(s, r, o, y, m, d, s_t, s_e, o_t, o_e).view(-1, self.args.nneg + 1)
                    ls = ls_f(sc, T.zeros(sc.size(0)).long().to(self.args.dvc))
                    ls.backward()
                    opt.step()
                    tot_ls += ls.item()

                    self.tb_sw.add_scalar(f'train/loss/{e}', ls.item(), i)
                    pb.set_postfix(loss=f'{tot_ls / i:.6f}')
                    pb.update()

            self.save_mem()
            self.log_tensorboard('train', {'loss': tot_ls / len(self.dl)}, e)
            logging.info(f'epoch {e}/{self.args.ne}: loss={tot_ls / len(self.dl):.6f}')

            if self.args.vd and (e % self.args.vd_stp == 0 or e == self.args.ne):
                mtrs = self.test('vd')
                self.save_mem()
                self.log_tensorboard('valid', mtrs, e)
                logging.info(f'epoch {e}/{self.args.ne} Validation: {mtrs}')
                if self.mtrs.cnt == 0 or getattr(mtrs, self.args.mtr) > getattr(self.mtrs, self.args.mtr):
                    self.mtrs = mtrs
                    self.save(opt)

        self.tb_sw.add_hparams(vars(self.args), dict(self.mtrs))

    def test(self, chk):
        logging.info(f'started loading {chk} chunk')
        with mp.Pool(os.cpu_count()) as p:
            x_ts = p.map(self.ds.prepare, self.ds.chk[chk][:3])
        self.save_mem()
        logging.info(f'finished loading {chk} chunk')

        self.mdl.eval()
        mtrs = ut.Metric()
        with tqdm(total=len(self.ds.chk[chk]), desc='valid' if chk == 'vd' else 'test') as pb:
            for x in x_ts:
                s, r, o, y, m, d, s_t, s_e, o_t, o_e = ut.to(self.args.dvc, x)
                sc = self.mdl(s, r, o, y, m, d, s_t, s_e, o_t, o_e).detach().cpu().numpy()
                rk = (sc > sc[0]).sum() + 1
                mtrs.update(rk)
            pb.set_postfix(**dict(mtrs))
            pb.update()
        return mtrs

    def load(self, opt=None):
        chk = T.load(os.path.join(self.args.pth, f'chk_{self.args.mtr}.dat'))
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
        with open(self.args.mem_pth + '_tmp', 'wb') as f:
            pickle.dump(self.mem, f, protocol=pickle.HIGHEST_PROTOCOL)
        shutil.copy(self.args.mem_pth + '_tmp', self.args.mem_pth)
