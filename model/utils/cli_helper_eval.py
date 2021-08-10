'''
Author: your name
Date: 2021-08-03 16:30:37
LastEditTime: 2021-08-10 13:31:35
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /lanenet-lane-detection-pytorch/model/utils/cli_helper_eval.py
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
    parser.add_argument("--model",
                        help="Model path",
                        default='./log/best_model.pth')
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
    parser.add_argument("--save",
                        help="Directory to save output",
                        default="./test_output")
    return parser.parse_args()
