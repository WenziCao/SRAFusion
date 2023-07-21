import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from src.datasets.build import build_MSRS_dataloader
from src.models.build import build_vsg_net

from src.logger.build import build_logger
from src.utils.tool import create_file
from src.trainer.trainer_VSeg import train_one_epoch

from src.runner.registry import runner


@runner.register
def train_vsg(cfg):
    # Instantiate the SummaryWriter object
    print('Start Tensorboard with "tensorboard --logdir runs", view at http://localhost:6006/')
    now = time.localtime()
    nowt = time.strftime("%Y-%m-%d-%H_%M_%S", now)
    vision_file_path = r'./visualization/runs/'
    create_file(vision_file_path)
    vision_file_path = r'./visualization/runs/' + cfg.MODEL.SEG_NET.TYPEV
    create_file(vision_file_path)
    vision_file_path = os.path.join(vision_file_path, nowt)
    create_file(vision_file_path)
    tb_writer = SummaryWriter(log_dir=vision_file_path)

    # dataloader
    train_loader = build_MSRS_dataloader(cfg)

    # model
    vi_seg_net = build_vsg_net(cfg)
    vi_seg_net.train()
    vi_seg_net.to(cfg.DEVICE)

    pg = [p for p in vi_seg_net.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=cfg.TRAINER.OPTIM.LR_START, momentum=0.9, weight_decay=0.0005)

    # Whether to restart training from the last breakpoint
    if cfg.TRAINER.RESUME:
        save_cp_dir = os.path.join('./model_hub/', cfg.MODEL.SEG_NET.TYPEV)
        create_file(save_cp_dir)
        saved_model_path = os.path.join(save_cp_dir + '/',
                                        'cp-epoch-{}.pt'.format(cfg.TRAINER.START_EPOCH-1))
        checkpoint = torch.load(saved_model_path)
        vi_seg_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # logger
    logger = build_logger(cfg)
    logger.info('start training......\n')

    train_begin_time = time.time()
    for epoch in range(cfg.TRAINER.START_EPOCH, cfg.TRAINER.EPOCH+1):

        mean_loss = train_one_epoch(cfg, vi_seg_net, optimizer, train_loader, epoch, logger)
        logger.info('epoch %d train mean loss: %.4f \n' % (epoch, mean_loss))
        if epoch % cfg.TRAINER.SV_ER_EP == 0:
            checkpoint = {
                'model_state_dict': vi_seg_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            save_cp_dir = os.path.join('./model_hub/', cfg.MODEL.SEG_NET.TYPEV)
            create_file(save_cp_dir)
            torch.save(checkpoint, os.path.join(save_cp_dir, 'cp-epoch-%d.pt' % epoch))
            logger.info('save model %d successfully......\n' % epoch)

        tags = ["mean_loss", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)

    train_end_time = time.time()
    train_total_time = train_end_time - train_begin_time
    logger.info('total training time is {}'.format(train_total_time))


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict
    with open('../../config/train/pretrain_vseg.yaml') as f:
        _cfg = yaml.safe_load(f)
    train_vsg(EasyDict(_cfg))