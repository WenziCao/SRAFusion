import time

import torch
from torch.autograd import Variable
from src.losses.build import build_seg_loss


def train_one_epoch(cfg, vi_seg_net, optimizer, data_loader, epoch, logger):

    lr_this_epo = cfg.TRAINER.OPTIM.LR_START * cfg.TRAINER.OPTIM.LR_DECAY ** (epoch - 1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_epo

    loss_function_seg = build_seg_loss(cfg)

    mean_loss = torch.zeros(1).to(cfg.DEVICE)

    max_step = len(data_loader)

    logger.info('epoch {} begin'.format(epoch))

    for step, data in enumerate(data_loader):
        start_time = time.time()
        image_vi, _, labels, _ = data
        image_vis = Variable(image_vi).to(cfg.DEVICE)
        label = Variable(labels).to(cfg.DEVICE)

        optimizer.zero_grad()

        logits = vi_seg_net(image_vis)
        lb = torch.squeeze(label, 1)

        # seg loss
        loss_seg = loss_function_seg(logits, lb)

        # total loss for Net
        loss_total = loss_seg

        loss_total.backward()
        optimizer.step()

        end_time = time.time()
        step_time = end_time - start_time

        mean_loss = (mean_loss * step + loss_total.detach()) / (step + 1)

        lr = optimizer.param_groups[0]["lr"]
        message = ','.join(['it: {it}/{max_step}',
                            'lr: {lr:4f}',
                            'loss_mean:{loss_mean:.4f}',
                            'loss_seg_vi:{loss_seg:.4f}',
                            'step_time:{step_time:.4f}',
                            ])\
            .format(it=step+1, max_step=max_step, lr=lr, loss_mean=mean_loss.item(),
                    loss_seg=loss_seg.item(), step_time=step_time)
        logger.info(message)

    logger.info('epoch {} finish\n'.format(epoch))

    return mean_loss.item()

