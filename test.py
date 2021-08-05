'''
Author: your name
Date: 2021-08-03 16:30:37
LastEditTime: 2021-08-05 16:14:37
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /undefined/Users/wk/Desktop/Mac_Workspaces/lanenet-lane-detection-pytorch/test.py
'''
import time
import os
import sys

import torch
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from model.utils.cli_helper_test import parse_args
from model.utils.postprocess import embedding_post_process
import numpy as np
from PIL import Image
import pandas as pd
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_test_data(img_path, transform):
    img = Image.open(img_path)
    img = transform(img)
    return img


def test():
    if os.path.exists('test_output') == False:
        os.mkdir('test_output')
    args = parse_args()
    img_path = args.img
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    model = LaneNet(arch=args.model_type)
    if DEVICE == 'cuda:0':
        state_dict = torch.load(model_path)  #  默认保存的模型是gpu
    else:
        state_dict = torch.load(model_path,
                                map_location=torch.device('cpu'))  # cpu推理
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    outputs = model(dummy_input)

    input = Image.open(img_path)
    input = input.resize((resize_width, resize_height))
    input = np.array(input)

    instance_pred = torch.squeeze(
        outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
    binary_pred = torch.squeeze(
        outputs['binary_seg_pred']).to('cpu').numpy() * 255

    # postprocess
    seg_img = np.zeros_like(input)
    embedding = instance_pred.transpose((1, 2, 0))
    bandwidth = 1.5
    # use meanshift
    lane_seg_img = embedding_post_process(embedding, binary_pred, bandwidth, 2)
    #  print(lane_seg_img)
    #  print(lane_seg_img.shape)
    color = np.array([[255, 255, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]],
                     dtype='uint8')

    for i, lane_idx in enumerate(np.unique(lane_seg_img)):
        if lane_idx == 0:
            continue
        seg_img[lane_seg_img == lane_idx] = color[i - 1]
    img = cv2.addWeighted(src1=seg_img,
                          alpha=0.8,
                          src2=input,
                          beta=1,
                          gamma=0.)
    #  print(input.shape)
    cv2.imwrite('./test_output/demo_result.png', img)

    cv2.imwrite(os.path.join('test_output', 'input.jpg'), input)
    cv2.imwrite(os.path.join('test_output', 'instance_output.jpg'),
                instance_pred.transpose((1, 2, 0)))
    cv2.imwrite(os.path.join('test_output', 'binary_output.jpg'), binary_pred)


if __name__ == "__main__":
    test()