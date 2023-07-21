from torch.utils.data import DataLoader

from src.datasets.dataset import *


def build_MSRS_dataloader(cfg):

    MSRS_dataset = dataset[cfg.DATASET.NAME](
        dataset_split=cfg.DATASET.DATASET_SPLIT,
        data_dir_vis=cfg.DATASET.DATA_DIR_VIS,
        data_dir_ir=cfg.DATASET.DATA_DIR_IR,
        data_dir_label=cfg.DATASET.DATA_DIR_LABEL,
    )
    MSRS_loader = DataLoader(
        dataset=MSRS_dataset,
        batch_size=cfg.DATASET.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        drop_last=cfg.DATASET.DROP_LAST,
    )

    return MSRS_loader


def build_LLVIP_dataloader(cfg):
    LLVIP_dataset = dataset[cfg.DATASET.NAME](
        dataset_split=cfg.DATASET.DATASET_SPLIT,
        data_dir_vis=cfg.DATASET.DATA_DIR_VIS,
        data_dir_ir=cfg.DATASET.DATA_DIR_IR,
    )
    LLVIP_loader = DataLoader(
        dataset=LLVIP_dataset,
        batch_size=cfg.DATASET.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        drop_last=cfg.DATASET.DROP_LAST,
    )

    return LLVIP_loader


if __name__ == '__main__':
    import torch
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    from easydict import EasyDict


    def test_dataloader(dataloader):
        # Get a batch of data
        images_vis, images_ir, lbs, _ = next(iter(dataloader))
        # images_vis, images_ir, _ = next(iter(dataloader))

        # Visualize the data
        grid_vis = make_grid(images_vis, nrow=8, normalize=True)
        grid_ir = make_grid(images_ir, nrow=8, normalize=True)

        # Display images and labels
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(grid_vis.permute(1, 2, 0))
        plt.title('Visible Images')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(grid_ir.permute(1, 2, 0))
        plt.title('Infrared Images')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        print('Labels:', lbs)


    # Test with MSRS dataset
    cfg_MSRS = EasyDict({
        'DATASET': {
            'TYPE': 'MSRSDataSet',
            'DATASET_SPLIT': 'train',
            'DATA_DIR_VIS': r'..\..\data\MSRS\Visible\train\MSRS',
            'DATA_DIR_IR': r'..\..\data\MSRS\Infrared\train\MSRS',
            'DATA_DIR_LABEL': r'..\..\data\MSRS\Label\train\MSRS',
            'BATCH_SIZE': 16,
            'NUM_WORKERS': 1
        }
    })

    MSRS_dataloader = build_MSRS_dataloader(cfg_MSRS)
    test_dataloader(MSRS_dataloader)

    # Test with LLVIP dataset
    cfg_LLVIP = EasyDict({
        'DATASET': {
            'TYPE': 'LLVIPDataSet',
            'DATASET_SPLIT': 'train',
            'DATA_DIR_VIS': r'..\..\data\MSRS\Visible\train\MSRS',
            'DATA_DIR_IR': r'..\..\data\MSRS\Infrared\train\MSRS',
            'BATCH_SIZE': 16,
            'NUM_WORKERS': 1
        }
    })

    LLVIP_dataloader = build_LLVIP_dataloader(cfg_LLVIP)
    test_dataloader(LLVIP_dataloader)

