'''
Author: your name
Date: 2021-08-03 16:30:37
LastEditTime: 2021-08-23 16:33:33
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /lanenet-lane-detection-pytorch/model/utils/cli_helper_test.py
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", help="Img path")
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
    parser.add_argument('--src_dir',
                        type=str,
                        default='./data/test_img/',
                        help='The path of image to be tested')
    parser.add_argument("--backend",
                        help="backbone network",
                        default='resnet101')
    parser.add_argument("-o",
                        "--output_path",
                        default="./test_output/",
                        help="path to output video")
    parser.add_argument("-r",
                        "--rtsp_url",
                        default="rtsp://192.168.31.70:8554/mystream")
    return parser.parse_args()
