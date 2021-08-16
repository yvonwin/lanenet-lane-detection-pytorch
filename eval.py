'''
Author: your name
Date: 2021-08-03 16:30:37
LastEditTime: 2021-08-16 15:58:30
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /lanenet-lane-detection-pytorch/eval.py
'''
import time
import os
# import sys
import torch
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
# from torch.autograd import Variable

from torchvision import transforms

from model.utils.cli_helper_eval import parse_args
from model.eval_function import Eval_Score
from  local_utils import  init_logger

# import numpy as np
# from PIL import Image
# import pandas as pd
# import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluation():
    args = parse_args()
    resize_height = args.height
    resize_width = args.width
    LOG = init_logger.get_logger(log_file_name_prefix='lanenet_eval')
    LOG.info('Evaluation Mode')
    LOG.info('dataset is %s' % args.dataset)
    LOG.info('model is %s' % args.model)
    LOG.info('backbone is %s' % args.model_type)
    LOG.info('backend is %s' % args.backend)

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
    model = LaneNet(arch=args.model_type, backend=args.backend)
    if DEVICE == 'cuda:0':
        LOG.info('Using GPU')
        state_dict = torch.load(model_path)
    else:
        LOG.info('Using CPU')
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
            LOG.info('单张推理时间为%s秒' % str(infer_end_time - infer_start_time))
            y_pred = torch.squeeze(y['binary_seg_pred'].to('cpu')).numpy()
            y_true = torch.squeeze(target).numpy()
            Score = Eval_Score(y_pred, y_true)
            dice += Score.Dice()
            iou += Score.IoU()
    end_time = time.time()
    # print(end_time - start_time)
    # print('Final_IoU: %s' % str(iou / len(eval_dataloader.dataset)))
    # print('Final_F1: %s' % str(dice / len(eval_dataloader.dataset)))
    LOG.info('use time: %s' % str(end_time - start_time))
    LOG.info('Final_IoU: %s' % str(iou / len(eval_dataloader.dataset)))
    LOG.info('Final_F1: %s' % str(dice / len(eval_dataloader.dataset)))


if __name__ == "__main__":

    evaluation()
