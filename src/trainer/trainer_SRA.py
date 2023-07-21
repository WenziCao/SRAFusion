import os
import time

import torch
from torch.autograd import Variable
from src.losses.build import build_fusion_loss

from src.utils.tool import RGB2YCrCb


def train_one_epoch(cfg, fs_net, vi_seg_net, ir_seg_net, optimizer, data_loader, epoch, logger):

    lr_this_epo = cfg.TRAINER.OPTIM.FS_LR_START * cfg.TRAINER.OPTIM.LR_DECAY ** (epoch - 1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_epo

    loss_function_fusion = build_fusion_loss(cfg)

    mean_loss = torch.zeros(1).to(cfg.DEVICE)

    max_step = len(data_loader)

    logger.info('epoch {} begin'.format(epoch))

    for step, data in enumerate(data_loader):
        start_time = time.time()
        image_vi, image_ir, labels, names = data
        image_vis = Variable(image_vi).to(cfg.DEVICE)
        image_vis_ycrcb = RGB2YCrCb(image_vis)
        image_ir = Variable(image_ir).to(cfg.DEVICE)
        labels = torch.unsqueeze(labels, dim=1)
        labels = Variable(labels).to(cfg.DEVICE)

        with torch.no_grad():
            logit_vi = vi_seg_net(image_vis)
            logit_v = logit_vi.argmax(1, keepdim=True)
            logit_ir = ir_seg_net(image_ir)
            logit_i = logit_ir.argmax(1, keepdim=True)

        optimizer.zero_grad()

        fused_v_i = fs_net(image_vis_ycrcb, image_ir, names, epoch)
        # total fusion loss and loss_in ,loss_grad
        loss_fusion, loss_bi, loss_uni, loss_lac = loss_function_fusion(image_vis_ycrcb, image_ir, fused_v_i,
                                                                        logit_v, logit_i, labels, names,
                                                                        epoch, cfg.TRAINER.SAVE_PIC)
        # total loss for Net
        loss_total = loss_fusion

        loss_total.backward()
        optimizer.step()

        end_time = time.time()
        step_time = end_time - start_time

        mean_loss = (mean_loss * step + loss_total.detach()) / (step + 1)

        lr = optimizer.param_groups[0]["lr"]
        message = ','.join(['it: {it}/{max_step}',
                            'lr: {lr:4f}',
                            'loss_mean:{loss_mean:.4f}',
                            'loss_fusion:{loss_fusion:.4f}',
                            'loss_bi:{loss_bi:.4f}',
                            'loss_uni:{loss_uni:.4f}',
                            'loss_lac:{loss_lac:.4f}',
                            'step_time:{step_time:.4f}',
                            ])\
            .format(it=step+1, max_step=max_step, lr=lr, loss_mean=mean_loss.item(),
                    loss_fusion=loss_fusion.item(), loss_bi=loss_bi.item(),
                    loss_uni=loss_uni.item(), loss_lac=loss_lac.item(), step_time=step_time)
        logger.info(message)

    logger.info('epoch {} finish\n'.format(epoch))

    return mean_loss.item()

