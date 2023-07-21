from src.models.nets.FusionNet import *
from src.models.nets.RGB_seg import *
from src.models.nets.T_seg import *


def build_fusion_net(cfg):
    fusion_net_instance = fusion_net[cfg.MODEL.FUSION_NET.NAME](
        output=cfg.MODEL.FUSION_NET.OUTPUT,
        save_fea_list=cfg.MODEL.FUSION_NET.SAVE_FEA_LIST
    )
    return fusion_net_instance


def build_vsg_net(cfg):
    seg_net_instance = seg_net[cfg.MODEL.SEG_NET.V_NAME](
        seg_model=cfg.MODEL.SEG_NET.BACKBONE,
        num_classes=cfg.MODEL.SEG_NET.NUM_CLASSES,
        output_stride=cfg.MODEL.SEG_NET.OUTPUT_STRIDE,
        separable_conv=cfg.MODEL.SEG_NET.SEPARABLE_CONV,
        pretrained_backbone=cfg.MODEL.SEG_NET.PRETRAINED_BACKBONE
    )
    return seg_net_instance


def build_isg_net(cfg):
    seg_net_instance = seg_net[cfg.MODEL.SEG_NET.I_NAME](
        seg_model=cfg.MODEL.SEG_NET.BACKBONE,
        num_classes=cfg.MODEL.SEG_NET.NUM_CLASSES,
        output_stride=cfg.MODEL.SEG_NET.OUTPUT_STRIDE,
        separable_conv=cfg.MODEL.SEG_NET.SEPARABLE_CONV,
        pretrained_backbone=cfg.MODEL.SEG_NET.PRETRAINED_BACKBONE
    )
    return seg_net_instance


if __name__ == '__main__':
    # Test the build_fusion_net function
    # You can replace the placeholders with actual values or modify the code according to your needs
    from easydict import EasyDict

    cfg_test = EasyDict({
        "MODEL": {
            "FUSION_NET": {
                "TYPE": 'FusionNet',
                "OUTPUT": 1,
                "SAVE_FEA_LIST": ["v_1", "f_4", "f_2"]
            },
            "SEG_NET": {
                "TYPEV": 'RgbSeg',
                "BACKBONE": 'deeplabv3plus_resnet101',
                "NUM_CLASSES": 10,
                "OUTPUT_STRIDE": 16,
                "SEPARABLE_CONV": False,
                "PRETRAINED_BACKBONE": False
            }
        }
    })

    fusnet = build_fusion_net(cfg_test)
    print(fusnet)
    sgnt = build_vsg_net(cfg_test)
    print(sgnt)

