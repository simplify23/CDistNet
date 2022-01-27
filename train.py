import os
import shutil
import argparse
import logging
# os.environ['CUDA_VISIBLE_DEVICES']="7"
from mmcv import Config
from thop import profile
import torch
from torch import optim
import torch.nn as nn
import torch.distributed as dist

from cdistnet.model.model import build_CDistNet
from cdistnet.data.data import make_data_loader, MyConcatDataset
# from cdistnet.data.hdf5loader import make_data_loader
from cdistnet.engine.trainer import do_train
from cdistnet.optim.optim import ScheduledOptim, WarmupOptim
from cdistnet.utils.tensorboardx import TensorboardLogger

def parse_args():
    parser = argparse.ArgumentParser(description='Train CDistNet')
    parser.add_argument('--config', type=str, default = 'configs/config.py',help='train config file path')
    parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    # assert not os.path.exists(cfg.train.model_dir), "{} already exists".format(cfg.train.model_dir)
    if not os.path.exists(cfg.train.model_dir):
        os.makedirs(cfg.train.model_dir)
    shutil.copy(args.config, cfg.train.model_dir)
    return cfg,args

def get_parameter_number(net):
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return 'Trainable: {} M'.format(trainable_num/1000000)

def get_flop_param(net):
    image=torch.randn(1, 3, 32, 96)
    tgt = torch.rand(1,180).long()
    flops, params = profile(net, inputs=(image, tgt))
    return 'Param: {} M, \n  FLOPS: {} G'.format(params/1000000,flops/1000000000)

def getlogger(mode_dir):
    logger = logging.getLogger('CDistNet')
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(mode_dir, 'log.txt' ))
    handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger
def train(cfg,args):
    # init dist_train
    if cfg.train_method=='dist':
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

    model = build_CDistNet(cfg)
    train_dataloader = make_data_loader(cfg, is_train=True)
    val_dataloader = []
    for val_gt_file in cfg.val.gt_file:
        val_dataloader.append(make_data_loader(cfg, is_train=False,val_gt_file=val_gt_file))
    n_current_steps = 0
    current_epoch = 0
    if cfg.train.model:
        model.load_state_dict(torch.load(cfg.train.model))
        current_epoch = cfg.train.current_epoch
        n_current_steps = current_epoch * len(train_dataloader)

    parse_nums = get_parameter_number(model)
    # parse_nums2 = get_flop_param(model)
    if cfg.train_method=='dist':
        model.cuda(args.local_rank)
        # 同步bn 可以验证准确性
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model,torch.distributed.new_group(ranks=[0]))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=True)
    else:
        model = nn.DataParallel(model)
        # model = nn.DataParallel(model,device_ids=[cfg.train.gpu_device_ids])
        device = torch.device(cfg.train.device)
        model = model.to(device)
        # print("device_count :{}".format(torch.cuda.device_count()))
    logger = getlogger(cfg.train.model_dir)
    if cfg.optim == 'warmup':
        logger.info("use warmup\n")
        optimizer = WarmupOptim(
            optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                betas=(0.9, 0.98),
                eps=1e-09,
            ),
            cfg.hidden_units, cfg.train.learning_rate_warmup_steps, n_current_steps,current_epoch)
    else:
        optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                betas=(0.9, 0.98),
                eps=1e-09,
            ),
            cfg.hidden_units, cfg.train.learning_rate_warmup_steps, n_current_steps)
    # optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)


    meter = TensorboardLogger(cfg.train.model_dir)
    logger.info("model parameter:-------\n{}".format(parse_nums))
    logger.info("model struct:-------\n{}".format(model))
    # logger.info("model compute:-------\n{}".format(parse_nums2)):
    # logger.info("step1:pos: {}, feat: {}, sem: {}\n".format(cfg.step1[0], cfg.step1[1], cfg.step1[2]))
    # logger.info("step2:feat_sem: {},pos_feat: {},pos_sem: {}\n".format(cfg.step2[0], cfg.step2[1], cfg.step2[2]))
    do_train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device = args.local_rank if cfg.train_method=='dist' else device,
        num_epochs=cfg.train.num_epochs,
        current_epoch=current_epoch,
        logger=logger,
        meter=meter,
        save_iter=cfg.train.save_iter,
        display_iter=cfg.train.display_iter,
        tfboard_iter=cfg.train.tfboard_iter,
        eval_iter=cfg.train.eval_iter,
        model_dir=cfg.train.model_dir,
        label_smoothing=cfg.train.label_smoothing,
        grads_clip=cfg.train.grads_clip,
        cfg=cfg,
    )


def main():
    cfg,args = parse_args()
    train(cfg,args)


if __name__ == '__main__':
    main()
