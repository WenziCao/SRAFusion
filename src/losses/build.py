from src.losses.fuse_loss import *
from src.losses.seg_loss import *


def build_fusion_loss(cfg):
    fusion_loss_instance = fuseloss[cfg.LOSS.FUSION_LOSS.NAME](
    )
    return fusion_loss_instance


def build_seg_loss(cfg):
    seg_loss_instance = segloss[cfg.LOSS.SEG_LOSS.NAME](
        thresh=cfg.LOSS.SEG_LOSS.THRESH,
        n_min=cfg.LOSS.SEG_LOSS.N_MIN,
        ignore_lb=cfg.LOSS.SEG_LOSS.IGNORE_LB,
    )
    return seg_loss_instance


if __name__ == '__main__':
    from easydict import EasyDict
    cfg_test = EasyDict({
        "LOSS": {
            "FUSION_LOSS": {
                "TYPE": 'SRAFLoss',
            },
            "SEG_LOSS": {
                "TYPE": 'OhemCELoss',
                "THRESH": 0.7,
                "N_MIN": 480 * 640,
            }
        }
    })
    f_ls = build_fusion_loss(cfg_test)
    print(f_ls)
    s_ls = build_seg_loss(cfg_test)
    print(s_ls)
