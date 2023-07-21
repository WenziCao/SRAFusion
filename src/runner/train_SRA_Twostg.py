import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import torch
from torch import optim
from torch.autograd import Variable

from src.datasets.build import build_MSRS_dataloader
from src.models.build import build_fusion_net
from src.models.build import build_vsg_net
from src.models.build import build_isg_net

from src.losses.build import build_fusion_loss
from src.losses.build import build_seg_loss
from src.logger.build import build_logger
from src.utils.tool import create_file, RGB2YCrCb, YCrCb2RGB

from src.runner.registry import runner


@runner.register
def train_sra_twostg(cfg):
    fs_net = build_fusion_net(cfg)
    vi_seg_net = build_vsg_net(cfg)
    ir_seg_net = build_isg_net(cfg)

    # model init
    pretrained_pth_fuse = os.path.join(r'./checkpoints/' + cfg.MODEL.FUSION_NET.OSTG_TYPE + '/cp-epoch-50.pt')
    cp_fuse = torch.load(pretrained_pth_fuse)
    fs_net.load_state_dict(cp_fuse['model_state_dict'])
    print('fuse init')

    pretrained_pth_vi = os.path.join(r'./model_hub/' + cfg.MODEL.SEG_NET.TYPEV + '/cp-epoch-50.pt')
    cp_vi = torch.load(pretrained_pth_vi)
    vi_seg_net.load_state_dict(cp_vi['model_state_dict'])
    print('vi_seg init')

    pretrained_pth_ir = os.path.join(r'./model_hub/' + cfg.MODEL.SEG_NET.TYPEI + '/cp-epoch-50.pt')
    cp_ir = torch.load(pretrained_pth_ir)
    ir_seg_net.load_state_dict(cp_ir['model_state_dict'])
    print('ir_seg init')

    # to device
    fs_net.to(cfg.DEVICE)
    vi_seg_net.to(cfg.DEVICE)
    ir_seg_net.to(cfg.DEVICE)
    # loss func
    fusion_loss = build_fusion_loss(cfg)
    loss_function_seg = build_seg_loss(cfg)

    # optimizer
    optimizer_fuse = torch.optim.Adam(fs_net.parameters(), lr=cfg.TRAINER.OPTIM.FS_LR_START)
    pg_vi = [p for p in vi_seg_net.parameters() if p.requires_grad]
    optimizer_vi = optim.SGD(pg_vi, lr=cfg.TRAINER.OPTIM.VSEG_LR_START, momentum=0.9, weight_decay=0.0005)
    # dataloader
    train_loader = build_MSRS_dataloader(cfg)
    # logger
    logger = build_logger(cfg)
    logger.info('start training......\n')

    for epoch in range(cfg.TRAINER.START_EPOCH, cfg.TRAINER.EPOCH+1):

        logger.info('epoch {} begin'.format(epoch))
        mean_loss_fusion = torch.zeros(1).to(cfg.DEVICE)
        mean_seg_vi_loss = torch.zeros(1).to(cfg.DEVICE)

        max_step = len(train_loader)

        for step, data in enumerate(train_loader):
            start_time = time.time()
            image_vi, image_ir, labels, names = data
            image_vis = Variable(image_vi).to(cfg.DEVICE)
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).to(cfg.DEVICE)
            labels = Variable(labels).to(cfg.DEVICE)

            loss_fusion, loss_bi, loss_uni, loss_lac, lr_fuse = train_fuse(cfg=cfg, fuse_model=fs_net, visegnet=vi_seg_net, irsegnet=ir_seg_net,
                                                                           optimizer_fuse=optimizer_fuse, image_vis=image_vis,
                                                                           image_vis_ycrcb=image_vis_ycrcb, image_ir=image_ir, labels=labels,
                                                                           names=names, fusion_loss=fusion_loss, epoch=epoch)

            loss_seg_vi, lr_seg = train_seg(cfg=cfg, fuse_model=fs_net, visegnet=vi_seg_net,
                                            optimizer_vi=optimizer_vi, image_vis_ycrcb=image_vis_ycrcb,
                                            image_ir=image_ir, labels=labels, names=names,
                                            loss_seg_vi=loss_function_seg, epoch=epoch)
            end_time = time.time()
            step_time = end_time - start_time

            mean_loss_fusion = (mean_loss_fusion * step + loss_fusion) / (step + 1)
            mean_seg_vi_loss = (mean_seg_vi_loss * step + loss_seg_vi) / (step + 1)

            message = ','.join(['it: {it}/{max_step}',
                                'lr_fuse: {lr_fuse:4f}',
                                'lr_seg: {lr_seg:4f}',
                                'mean_loss_fusion:{mean_loss_fusion:.4f}',
                                'mean_seg_vi_loss:{mean_seg_vi_loss:.4f}',
                                'loss_fusion:{loss_fusion:.4f}',
                                'loss_bi:{loss_bi:.4f}',
                                'loss_uni:{loss_uni:4f}',
                                'loss_lac:{loss_lac:4f}',
                                'step_time:{step_time:.4f}',
                                ]) \
                .format(it=step + 1, max_step=max_step, lr_fuse=lr_fuse, lr_seg=lr_seg,
                        mean_loss_fusion=mean_loss_fusion.item(), mean_seg_vi_loss=mean_seg_vi_loss.item(),
                        loss_fusion=loss_fusion.item(), loss_bi=loss_bi.item(),
                        loss_uni=loss_uni.item(), loss_lac=loss_lac.item(), step_time=step_time)
            logger.info(message)

        if epoch % cfg.TRAINER.SV_ER_EP == 0:
            cp_fuse_net = {
                'model_state_dict': fs_net.state_dict(),  # *模型参数
                # 'optimizer_state_dict': optimizer_fuse.state_dict(),  # *优化器参数
                }
            save_fuse_dir = os.path.join('./checkpoints/', cfg.MODEL.FUSION_NET.TSTG_TYPE)
            create_file(save_fuse_dir)
            torch.save(cp_fuse_net, os.path.join(save_fuse_dir, 'cp-epoch-%d.pt' % epoch))
            logger.info('save fuse model %d successfully......\n' % epoch)

            cp_seg_vi_net = {
                'model_state_dict': vi_seg_net.state_dict(),  # *模型参数
                # 'optimizer_state_dict': optimizer_vi.state_dict(),  # *优化器参数
                }
            save_vi_dir = os.path.join('./checkpoints/', cfg.MODEL.SEG_NET.TYPEV_LATE)
            create_file(save_vi_dir)
            torch.save(cp_seg_vi_net, os.path.join(save_vi_dir, 'cp-epoch-%d.pt' % epoch))
            logger.info('save vi seg model %d successfully......\n' % epoch)

        logger.info('epoch {} finish\n'.format(epoch))


def train_fuse(cfg, fuse_model, visegnet, irsegnet, optimizer_fuse, image_vis, image_vis_ycrcb,
               image_ir, labels, names, fusion_loss, epoch):

    lr_this_epo = cfg.TRAINER.OPTIM.FS_LR_START * cfg.TRAINER.OPTIM.LR_DECAY ** (epoch - 1)

    for param_group in optimizer_fuse.param_groups:
        param_group['lr'] = lr_this_epo
    fuse_model.train()
    labels = torch.unsqueeze(labels, dim=1)
    with torch.no_grad():
        logit_vi = visegnet(image_vis)
        logit_v = logit_vi.argmax(1, keepdim=True)
        logit_ir = irsegnet(image_ir)
        logit_i = logit_ir.argmax(1, keepdim=True)

    optimizer_fuse.zero_grad()
    fused_v_i = fuse_model(image_vis_ycrcb, image_ir, names, epoch)

    # fusion loss
    loss_fusion, loss_bi, loss_uni, loss_lac = fusion_loss(image_vis_ycrcb, image_ir, fused_v_i,
                                                           logit_v, logit_i, labels, names,
                                                           epoch, cfg.TRAINER.SAVE_PIC)

    loss_total = loss_fusion
    loss_total.backward()
    optimizer_fuse.step()

    return loss_fusion, loss_bi, loss_uni, loss_lac, lr_this_epo


def train_seg(cfg, fuse_model, visegnet, optimizer_vi, image_vis_ycrcb, image_ir,
              labels, names, loss_seg_vi, epoch):

    lr_this_epo = cfg.TRAINER.OPTIM.VSEG_LR_START * cfg.TRAINER.OPTIM.LR_DECAY ** (epoch - 1)
    for param_group in optimizer_vi.param_groups:
        param_group['lr'] = lr_this_epo

    visegnet.train()
    optimizer_vi.zero_grad()

    with torch.no_grad():
        fusion_v_i = fuse_model(image_vis_ycrcb, image_ir, names, epoch)
        fusion_ycrcb = torch.cat(
            (fusion_v_i, image_vis_ycrcb[:, 1:2, :, :], image_vis_ycrcb[:, 2:, :, :]), dim=1)
        fusion_image = YCrCb2RGB(fusion_ycrcb)
        ones = torch.ones_like(fusion_image)
        zeros = torch.zeros_like(fusion_image)
        fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
        fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)

    logit_vi = visegnet(fusion_image)
    lb = torch.squeeze(labels, 1)
    loss_vi_seg = loss_seg_vi(logit_vi, lb)

    loss_vi_seg.backward()
    optimizer_vi.step()

    return loss_vi_seg, lr_this_epo


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict
    with open('../../config/cfg.yaml') as f:
        _cfg = yaml.safe_load(f)
    train_sra_twostg(EasyDict(_cfg))
