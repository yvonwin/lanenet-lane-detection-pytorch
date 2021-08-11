'''
Author: your name
Date: 2021-08-03 16:30:37
LastEditTime: 2021-08-11 14:10:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /lanenet-lane-detection-pytorch/eval.py
'''
import time
import os
import sys

import torch
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader, dataloader
from torch.autograd import Variable

from torchvision import transforms

from model.utils.cli_helper_eval import parse_args
from model.eval_function import Eval_Score

import numpy as np
from PIL import Image
import pandas as pd
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluation():
    args = parse_args()
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    target_transforms = transforms.Compose([
        Rescale((resize_width, resize_height)),
    ])

    dataset_file = os.path.join(args.dataset, 'test.txt')
    Eval_Dataset = TusimpleSet(dataset_file,
                               transform=data_transform,
                               target_transform=target_transforms)
    eval_dataloader = DataLoader(Eval_Dataset, batch_size=1, shuffle=True)

    model_path = args.model
    model = LaneNet(arch=args.model_type)
    if DEVICE == 'cuda:0':
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    iou, dice = 0, 0
    start_time = time.time()
    with torch.no_grad():
        for x, target, _ in eval_dataloader:
            infer_start_time = time.time()
            y = model(x.to(DEVICE))
            infer_end_time = time.time()
            print('推理时间为%s秒' % str(infer_end_time - infer_start_time))
            y_pred = torch.squeeze(y['binary_seg_pred'].to('cpu')).numpy()
            y_true = torch.squeeze(target).numpy()
            Score = Eval_Score(y_pred, y_true)
            dice += Score.Dice()
            iou += Score.IoU()
    end_time = time.time()
    print(end_time - start_time)
    print('Final_IoU: %s' % str(iou / len(eval_dataloader.dataset)))
    print('Final_F1: %s' % str(dice / len(eval_dataloader.dataset)))


if __name__ == "__main__":

    evaluation()
