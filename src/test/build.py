from src.test.test_Fuse import *
from src.test.test_VSeg import *
from src.test.test_ISeg import *

import yaml
from easydict import EasyDict


def build_test_func(args):
    if args.mode == 'test_fuse':
        with open('./config/test/test_fuse.yaml') as f:
            _cfg = yaml.safe_load(f)
        tester[args.mode](EasyDict(_cfg))

    if args.mode == 'test_isg':
        with open('./config/test/test_iseg.yaml') as f:
            _cfg = yaml.safe_load(f)
        tester[args.mode](EasyDict(_cfg))

    if args.mode == 'test_vsg':
        with open('./config/test/test_vseg.yaml') as f:
            _cfg = yaml.safe_load(f)
        tester[args.mode](EasyDict(_cfg))