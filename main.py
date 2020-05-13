import logging
import os

import torch as T

from src import utils as ut
from src.dataset import Dataset
from src.runner import Runner


if __name__ == '__main__':
    args = ut.args()
    args.dvc = 'cuda' if T.cuda.is_available() else 'cpu'
    args.pth = f'models/{args.dataset}_{args.model}_{args.id}'

    os.makedirs(args.pth, exist_ok=True)
    ut.logger(args)

    for k, v in sorted(vars(args).items()):
        logging.info(f'{k} = {v}')

    r = Runner(Dataset(args.dataset), args)

    if args.tr:
        r.train()

    if args.ts:
        with T.no_grad():
            r.load()
            mtrs = r.test('ts')
            r.log_tensorboard('valid', mtrs, 0)
            logging.info(f'Test: {mtrs}')
