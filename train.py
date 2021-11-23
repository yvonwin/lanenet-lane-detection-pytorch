"""
Author: yvon
Date: 2021-08-03 16:30:37
LastEditTime: 2021-08-20 16:03:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /lanenet-lane-detection-pytorch/train.py
"""
import time
import os

# import sys

import torch
from model.lanenet.train_lanenet import train_model
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader

# from torch.autograd import Variable

from torchvision import transforms

from model.utils.cli_helper import parse_args

# from model.eval_function import Eval_Score

# import numpy as np
# import pandas as pd
from local_utils import init_logger

# import loguru
# import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train():
    """训练模型
    """
    args = parse_args()
    # 日志信息记录
    LOG = init_logger.get_logger(log_file_name_prefix="lanenet_train")
    LOG.info("use model_type %s" % args.model_type)
    LOG.info("use backend %s" % args.backend)
    LOG.info("use lr %s" % str(args.lr))
    LOG.info("use augment? %s" % str(args.use_aug))
    LOG.info("use dataset: %s" % args.dataset)
    LOG.info("log save_dir is: %s" % args.save)
    LOG.info("use loss_type %s" % args.loss_type)
    LOG.info("use batch  size is: %s" % args.bs)
    # LOG.info('set epochs %s' % str(args.epochs))
    LOG.info("resize image_width is:%s, image_height is:%s\n" % (args.width, args.height))
    save_path = args.save
    # 如果不存在则创建保存模型路径
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    train_dataset_file = os.path.join(args.dataset, "train.txt")
    val_dataset_file = os.path.join(args.dataset, "val.txt")

    resize_height = args.height
    resize_width = args.width

    # 数据预处理
    # 使用数据增强
    if args.use_aug:
        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.Resize((resize_height, resize_width)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    # 数据增强. TODO 需要测试
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize((resize_height, resize_width)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }
    # 不使用数据增强
    else:
        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.Resize((resize_height, resize_width)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize((resize_height, resize_width)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }

    target_transforms = transforms.Compose(
        [
            Rescale((resize_width, resize_height)),
        ]
    )

    # 训练数据加载
    train_dataset = TusimpleSet(
        train_dataset_file, transform=data_transforms["train"], target_transform=target_transforms
    )
    # train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=False
    )  # 如果内存够的话 , 可以开启pin_memory=True

    # 验证数据加载
    val_dataset = TusimpleSet(val_dataset_file, transform=data_transforms["val"], target_transform=target_transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True)

    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": len(train_loader.dataset), "val": len(val_loader.dataset)}
    # 加载模型
    model = LaneNet(arch=args.model_type, backend=args.backend)
    model.to(DEVICE)
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    LOG.info(f"{args.epochs} epochs {len(train_dataset)} training samples")

    # 设置scheduler，使用余弦退火算法
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones = [30,80], gamma=0.5, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #  mode='max', factor=0.5, patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, eta_min=1e-4)
    # 开始训练
    model = train_model(
        model,
        optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        device=DEVICE,
        loss_type=args.loss_type,
        num_epochs=args.epochs,
    )

    save_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    if args.model_type == "DeepLabv3+":
        model_save_filename = os.path.join(
            save_path,
            save_time + "_epochs" + str(args.epochs) + "_" + args.model_type + "_" + args.backend + "_best_model.pth",
        )
    else:
        model_save_filename = os.path.join(
            save_path, save_time + "_epochs" + str(args.epochs) + "_" + args.model_type + "_" + "_best_model.pth"
        )
    # 保存模型
    torch.save(model.state_dict(), model_save_filename)
    LOG.info("model is saved: {}".format(model_save_filename))
    LOG.info("Complete training process good luck!!")
    return


if __name__ == "__main__":
    train()
