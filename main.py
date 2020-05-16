import logging
import os

import torch as T

from src import utils as ut
from src.runner import Runner


if __name__ == '__main__':
    args = ut.args()
    args.dvc = 'cuda' if T.cuda.is_available() else 'cpu'
    args.pth = f'models/{args.dataset}_{args.model}_{args.id}'
    args.mem_pth = f'memory/{args.dataset}.pkl'

    os.makedirs(args.pth, exist_ok=True)
    ut.logger(args)

    for k, v in sorted(vars(args).items()):
        logging.info(f'{k} = {v}')

    r = Runner(args)

    if args.tr:
        r.train()

    if args.ts:
        r.test()
