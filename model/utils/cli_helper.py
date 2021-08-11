'''
Author: your name
Date: 2021-08-03 16:30:37
LastEditTime: 2021-08-11 11:22:54
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /lanenet-lane-detection-pytorch/model/utils/cli_helper.py
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        help="Dataset path",
                        default='./dataset_new')
    parser.add_argument("--model_type",
                        help="Model type",
                        default='DeepLabv3+')
    parser.add_argument("--loss_type", help="Loss type", default='FocalLoss')
    parser.add_argument("--save",
                        required=False,
                        help="Directory to save model",
                        default="./log")
    parser.add_argument("--epochs",
                        required=False,
                        type=int,
                        help="Training epochs",
                        default=25)
    parser.add_argument("--width",
                        required=False,
                        type=int,
                        help="Resize width",
                        default=512)
    parser.add_argument("--height",
                        required=False,
                        type=int,
                        help="Resize height",
                        default=256)
    parser.add_argument("--bs",
                        required=False,
                        type=int,
                        help="Batch size",
                        default=4)
    parser.add_argument("--val",
                        required=False,
                        type=bool,
                        help="Use validation",
                        default=False)
    parser.add_argument("--lr",
                        required=False,
                        type=float,
                        help="Learning rate",
                        default=0.0001)  # deeplabv3+原始是0.0007
    parser.add_argument("--pretrained",
                        required=False,
                        default=None,
                        help="pretrained model path")
    parser.add_argument("--image",
                        default="./output",
                        help="output image folder")
    parser.add_argument("--net", help="backbone network")
    parser.add_argument("--json", help="post processing json")
    parser.add_argument("--checkpoint", required=False, help="checkpoint path")
    parser.add_argument("--num_workers", required=False, default=4, type=int)
    return parser.parse_args()
