'''
Author: your name
Date: 2021-08-03 16:30:37
LastEditTime: 2021-08-17 18:14:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: ~/Mac_Workspaces/lanenet-lane-detection-pytorch/test.py
'''
# import time
import os
from sklearn import cluster
# import sys

import torch
# from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
from torchvision import transforms
from model.utils import postprocess
from model.utils.cli_helper_test import parse_args
from model.utils.postprocess import embedding_post_process
import numpy as np
from PIL import Image
# import pandas as pd
import cv2
from model.utils import lanenet_cluster, lanenet_postprocess

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_test_data(img_path, transform):
    img = Image.open(img_path)
    img = transform(img)
    return img


def test():
    if not os.path.exists('test_output'):
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
    model = LaneNet(arch=args.model_type, backend=args.backend)
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)  # 默认保存的模型是gpu
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

    # get instance and binary
    instance_pred = torch.squeeze(
        # outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
        outputs['instance_seg_logits'].detach().to('cpu')).numpy()
    binary_pred = torch.squeeze(
        # outputs['binary_seg_pred']).to('cpu').numpy() * 255
        outputs['binary_seg_pred']).to('cpu').numpy()



    # postprocess
    instance_pred = instance_pred.transpose(1, 2, 0)
    cluster = lanenet_cluster.LaneNetCluster()
    #postprocessor = lanenet_postprocess.LaneNetPoseProcessor()
    mask_image, _, _, _ = cluster.get_lane_mask(instance_seg_ret=instance_pred,
                                                binary_seg_ret=binary_pred,
                                                gt_image=input)
                                                
    cv2.imwrite('./test_output/mask_result.png', mask_image)
    cv2.imwrite(os.path.join('test_output', 'input.jpg'), input)
    # cv2.imwrite(os.path.join('test_output', 'instance_output.jpg'),
    #             instance_pred.transpose((1, 2, 0)))
    cv2.imwrite(os.path.join('test_output', 'instance_output.jpg'),
                instance_pred * 255)
    # cv2.imwrite(os.path.join('test_output', 'binary_output.jpg'),
    #  binary_pred)
    cv2.imwrite(os.path.join('test_output', 'binary_output.jpg'),
                binary_pred * 255)


if __name__ == "__main__":
    test()
