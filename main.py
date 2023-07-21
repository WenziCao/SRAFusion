import argparse
from src.runner.build import build_train_func
from src.test.build import build_test_func


def main(args):
    if args.train_or_test == 'train':
        build_train_func(args)
    elif args.train_or_test == 'test':
        build_test_func(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_or_test', '-tort', type=str, default='test', help='train or test')
    # 'train_vsg' 'train_isg' 'train_sra_onestg' 'train_sra_twostg' 'test_fuse' 'test_ISeg' 'test_VSeg'
    parser.add_argument('--mode', '-mode', type=str, default='test_fuse', help='Select the corresponding training or testing type')
    opt = parser.parse_args()
    main(opt)