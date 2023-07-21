from src.runner.train_ISeg import *
from src.runner.train_VSeg import *
from src.runner.train_SRA_Onestg import *
from src.runner.train_SRA_Twostg import *


import yaml
from easydict import EasyDict


def build_train_func(args):
    if args.mode == 'train_vsg':
        with open('./config/train/pretrain_vseg.yaml') as f:
            _cfg = yaml.safe_load(f)
        runner[args.mode](EasyDict(_cfg))

    elif args.mode == 'train_isg':
        with open('./config/train/pretrain_iseg.yaml') as f:
            _cfg = yaml.safe_load(f)
        runner[args.mode](EasyDict(_cfg))

    elif args.mode == 'train_sra_onestg':
        with open('./config/train/train_sra_1th_stage.yaml') as f:
            _cfg = yaml.safe_load(f)
        runner[args.mode](EasyDict(_cfg))

    elif args.mode == 'train_sra_twostg':
        with open('./config/train/train_sra_2th_stage.yaml') as f:
            _cfg = yaml.safe_load(f)
        runner[args.mode](EasyDict(_cfg))
