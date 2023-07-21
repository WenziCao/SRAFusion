import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

import torch
from PIL import Image
from torch.autograd import Variable

from src.datasets.build import build_MSRS_dataloader
from src.models.build import build_fusion_net
from src.utils.tool import RGB2YCrCb, YCrCb2RGB, create_file

from src.test.registry import tester


@tester.register
def test_fuse(cfg):
    # save fusion result path
    file_path = os.path.join('./output/' + cfg.MODEL.FUSION_NET.STG_TYPE)
    create_file(file_path)
    file_path = os.path.join(file_path + '/' + 'epoch_{}'.format(cfg.TEST_EPOCH))
    create_file(file_path)

    # saved model path
    save_model_path = os.path.join('./checkpoints/', cfg.MODEL.FUSION_NET.STG_TYPE,
                                   'cp-epoch-{}.pt'.format(cfg.TEST_EPOCH))

    # get the model
    model = build_fusion_net(cfg)
    model.to(cfg.DEVICE)
    model.eval()
    checkpoint = torch.load(save_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('model load done!')

    # dataset
    test_loader = build_MSRS_dataloader(cfg)
    test_loader.n_iter = len(test_loader)
    epoch = 0
    # begin to train
    with torch.no_grad():
        for it, (images_vis, images_ir, labels, name) in enumerate(test_loader):

            images_vis = Variable(images_vis).to(cfg.DEVICE)
            images_ir = Variable(images_ir).to(cfg.DEVICE)

            images_vis_ycrcb = RGB2YCrCb(images_vis)

            fused_y = model(images_vis_ycrcb, images_ir, name, epoch)
            # fuse part
            fusion_ycrcb = torch.cat(
                (fused_y, images_vis_ycrcb[:, 1:2, :, :], images_vis_ycrcb[:, 2:, :, :]), dim=1,)
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)
            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image))
            fused_image = np.uint8(255.0 * fused_image)
            # fused_dir = './Fusion_results/'
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = Image.fromarray(image)
                # fused image save file
                save_path = os.path.join(file_path, name[k])
                image.save(save_path)
                print('Fusion {0} finished!'.format(save_path))

