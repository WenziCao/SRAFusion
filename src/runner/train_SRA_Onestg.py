import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from src.datasets.build import build_MSRS_dataloader
from src.models.build import build_fusion_net
from src.models.build import build_vsg_net
from src.models.build import build_isg_net

from src.logger.build import build_logger
from src.utils.tool import create_file
from src.trainer.trainer_SRA import train_one_epoch

from src.runner.registry import runner


@runner.register
def train_sra_onestg(cfg):
    # Instantiate the SummaryWriter object
    print('Start Tensorboard with "tensorboard --logdir runs", view at http://localhost:6006/')
    now = time.localtime()
    nowt = time.strftime("%Y-%m-%d-%H_%M_%S", now)
    vision_file_path = r'./visualization/runs/'
    create_file(vision_file_path)
    vision_file_path = './visualization/runs/' + cfg.MODEL.FUSION_NET.OSTG_TYPE
    create_file(vision_file_path)
    vision_file_path = os.path.join(vision_file_path, nowt)
    create_file(vision_file_path)
    tb_writer = SummaryWriter(log_dir=vision_file_path)
    # dataloader
    train_loader = build_MSRS_dataloader(cfg)
    # fusion model

    fs_net = build_fusion_net(cfg)
    fs_net.train()
    fs_net.to(cfg.DEVICE)

    # vi seg model, load pretrained checkpoint
    vi_seg_net = build_vsg_net(cfg)
    vi_seg_net.eval()
    saved_vi_seg_path = os.path.join('./model_hub/', cfg.MODEL.SEG_NET.TYPEV,
                                     'cp-epoch-{}.pt'.format(cfg.TRAINER.PRETR_SEG_EP))
    cp_vi = torch.load(saved_vi_seg_path)
    vi_seg_net.load_state_dict(cp_vi['model_state_dict'])
    vi_seg_net.to(cfg.DEVICE)
    print('vi seg model load done!')

    # ir seg model
    ir_seg_net = build_isg_net(cfg)
    ir_seg_net.eval()
    save_ir_seg_path = os.path.join('./model_hub/', cfg.MODEL.SEG_NET.TYPEI,
                                    'cp-epoch-{}.pt'.format(cfg.TRAINER.PRETR_SEG_EP))
    cp_ir = torch.load(save_ir_seg_path)
    ir_seg_net.load_state_dict(cp_ir['model_state_dict'])
    ir_seg_net.to(cfg.DEVICE)
    print('ir seg model load done!')

    pg = [p for p in fs_net.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=cfg.TRAINER.OPTIM.FS_LR_START, momentum=0.9, weight_decay=0.0005)

    # Whether to restart training from the last breakpoint
    '''
    if cfg.TRAINER.RESUME:
        save_cp_dir = os.path.join('./checkpoints/', cfg.MODEL.FUSION_NET.OSTG_TYPE)
        create_file(save_cp_dir)
        saved_model_path = os.path.join(save_cp_dir + '/',
                                        'cp-epoch-{}.pt'.format(cfg.TRAINER.START_EPOCH-1))
        checkpoint = torch.load(saved_model_path)
        fs_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    '''
    # 写入日志
    logger = build_logger(cfg)
    logger.info('start training......\n')

    train_begin_time = time.time()
    for epoch in range(cfg.TRAINER.START_EPOCH, cfg.TRAINER.EPOCH+1):

        mean_loss = train_one_epoch(cfg, fs_net, vi_seg_net, ir_seg_net, optimizer, train_loader, epoch, logger)

        logger.info('epoch %d train mean loss: %.4f \n' % (epoch, mean_loss))
        if epoch % cfg.TRAINER.SV_ER_EP == 0:
            checkpoint = {
                'model_state_dict': fs_net.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                }
            save_cp_dir = os.path.join('./checkpoints/', cfg.MODEL.FUSION_NET.OSTG_TYPE)
            create_file(save_cp_dir)
            torch.save(checkpoint, os.path.join(save_cp_dir, 'cp-epoch-%d.pt' % epoch))
            logger.info('save model %d successfully......\n' % epoch)

        tags = ["mean_loss", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)

    train_end_time = time.time()
    train_total_time = train_end_time - train_begin_time
    logger.info('total training time is {}'.format(train_total_time))

