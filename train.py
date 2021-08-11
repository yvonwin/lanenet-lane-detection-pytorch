'''
Author: your name
Date: 2021-08-03 16:30:37
LastEditTime: 2021-08-11 14:00:58
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /lanenet-lane-detection-pytorch/train.py
'''
# import time
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
import pandas as pd
# import cv2

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train():
    args = parse_args()
    save_path = args.save
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    train_dataset_file = os.path.join(args.dataset, 'train.txt')
    val_dataset_file = os.path.join(args.dataset, 'val.txt')

    resize_height = args.height
    resize_width = args.width

    # 数据预处理
    data_transforms = {
        'train':
        transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ColorJitter(brightness=0.1,
                                   contrast=0.1,
                                   saturation=0.1,
                                   hue=0.1),
            # TODO数据增强 需要测试
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val':
        transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    target_transforms = transforms.Compose([
        Rescale((resize_width, resize_height)),
    ])

    # 训练数据加载
    train_dataset = TusimpleSet(train_dataset_file,
                                transform=data_transforms['train'],
                                target_transform=target_transforms)
    # train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.bs,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=False)  # 如果内存够的话 , 可以开启pin_memory=True

    # 验证数据加载
    val_dataset = TusimpleSet(val_dataset_file,
                              transform=data_transforms['val'],
                              target_transform=target_transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True)

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {
        'train': len(train_loader.dataset),
        'val': len(val_loader.dataset)
    }

    # 加载模型
    model = LaneNet(arch=args.model_type)
    model.to(DEVICE)
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"{args.epochs} epochs {len(train_dataset)} training samples\n")

    # 设置scheduler. ready to test
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones = [30,80], gamma=0.9, last_epoch=-1)
    # 开始训练
    model, log = train_model(model,
                             optimizer,
                             scheduler=None,
                             dataloaders=dataloaders,
                             dataset_sizes=dataset_sizes,
                             device=DEVICE,
                             loss_type=args.loss_type,
                             num_epochs=args.epochs,
                             checkpoint=args.checkpoint)

    # 日志
    df = pd.DataFrame({'epoch': [], 'training_loss': [], 'val_loss': []})
    df['epoch'] = log['epoch']
    df['training_loss'] = log['training_loss']
    df['val_loss'] = log['val_loss']

    train_log_save_filename = os.path.join(save_path, 'training_log.csv')
    df.to_csv(train_log_save_filename,
              columns=['epoch', 'training_loss', 'val_loss'],
              header=True,
              index=False,
              encoding='utf-8')
    print("training log is saved: {}".format(train_log_save_filename))

    # 保存模型 训练完保存模型。
    model_save_filename = os.path.join(save_path, 'best_model.pth')
    torch.save(model.state_dict(), model_save_filename)
    print("model is saved: {}".format(model_save_filename))


if __name__ == '__main__':
    train()
