'''
Author: your name
Date: 2021-07-19 14:26:06
LastEditTime: 2021-08-23 18:17:04
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /refact_7.13/pipeline.py
'''
import os
import cv2
import numpy as np
import math
# import argparse
from RTSCapture import RTSCapture
import os.path as ops
# import time
# from sklearn import cluster
import torch
from model.lanenet.LaneNet import LaneNet
from torchvision import transforms
from model.utils.cli_helper_test import parse_args
# from model.utils.postprocess import embedding_post_process
from PIL import Image
import glob
from model.utils import lanenet_cluster, lanenet_postprocess
import matplotlib.pyplot as plt

FPS = 25
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def load_test_data(img_path, transform):
#     img = Image.open(img_path)
#     img = transform(img)
#     return img


def load_model(model_path, model_type, backend):
    model = LaneNet(arch=model_type, backend=backend)
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)  # 默认保存的模型是gpu
    else:
        state_dict = torch.load(model_path,
                                map_location=torch.device('cpu'))  # cpu推理
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    return model


def minmax_scale(input_arr):
    """
    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet_one_img(model, frame):

    """"
    :param src_dir:
    :param weights_path:
    :return:
    """
    args = parse_args()
    resize_height = args.height
    resize_width = args.width
    save_dir = args.save

    os.makedirs(save_dir, exist_ok=True)
    data_transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    dummy_input = data_transform(input)
    dummy_input = dummy_input.to(DEVICE)

    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    outputs = model(dummy_input)

    # input = Image.open(img_path)
    input = input.resize((resize_width, resize_height))
    input = np.array(input)

    instance_pred = torch.squeeze(
        outputs['instance_seg_logits'].detach().to('cpu')).numpy()
    binary_pred = torch.squeeze(
        outputs['binary_seg_pred']).to('cpu').numpy()

    # 开始后处理
    instance_pred = instance_pred.transpose(1, 2, 0)
    cluster = lanenet_cluster.LaneNetCluster()
    # 删除一些比较小的联通区域
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()
    binary_pred = postprocessor.postprocess(binary_pred)
    # print('*****fuck! img_name is: ', img_name)
    mask_image, _, _, _ = cluster.get_lane_mask(
        instance_seg_ret=instance_pred,
        binary_seg_ret=binary_pred,
        gt_image=input)
    # mask_image = mask_image[:, :, (2, 1, 0)]
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR)
    # print(instance_pred.shape)
    # for i in range(3):
    #     instance_pred[:, :, i] = minmax_scale(instance_pred[:, :, i])
    # embedding_image = np.array(instance_pred, np.uint8)

    # cv2.imwrite('./mask_img.png', mask_image[0])
    # 拓展binary_pred通道 方便可视化
    # bin_image = np.expand_dims(binary_pred, axis=2)
    # bin_image = np.concatenate((bin_image, bin_image, bin_image), axis=-1)
    # # 结果可视化
    # out_all = np.vstack([
    #     np.hstack([
    #         cv2.cvtColor(input, cv2.COLOR_RGB2BGR),
    #         cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR),
    #     ]),
    #     np.hstack([instance_pred * 255, bin_image * 255])
    # ])
    # cv2.imwrite(os.path.join(save_dir, 'input_' + img_name), input)
    # cv2.imwrite(os.path.join(save_dir, 'result_' + img_name), mask_image)

    # cv2.imwrite(os.path.join(save_dir, 'instance_output' + img_name),
    #            instance_pred * 255)
    # cv2.imwrite(os.path.join(save_dir, 'binary_output' + img_name),
    #            binary_pred * 255)
    # cv2.imwrite(os.path.join(save_dir, 'out_all' + img_name), out_all)
    # plt.figure('out_all')
    # plt.imshow(mask_image[:, :, (2, 1, 0)])
    # plt.figure('embedding')
    # plt.imshow(embedding_image)
    # plt.show()
    return mask_image


def process_video(model, rtsp_url, output_path):
    rtscap = RTSCapture.create(rtsp_url)
    rtscap.start_read()
    # framewidth = rtscap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # frameheight = rtscap.get(
    #    cv2.CAP_PROP_FRAME_HEIGHT)  # 图像横轴中心点（宽度）   #图像纵轴中心点（高度）ßß
    # 帧处理代码写这
    # out_videofile = output_path + '_test.mp4'
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # output_video = cv2.VideoWriter(out_videofile, fourcc, FPS,
    #                               (int(framewidth), int(frameheight)))
    print('start to write video')
    while rtscap.isStarted():
        ok, frame = rtscap.read_latest_frame()  # read_latest_frame() 替代 read() 此时取到的为最新的帧
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        if not ok:
            continue
        # out = process_an_image(frame, framewidth, frameheight)
        out = test_lanenet_one_img(model, frame)
        # print(out)
        # cv2.namedWindow('out_img', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('out_img', 1024, 756)
        cv2.imshow('out_img', out)
        # cv2.namedWindow('out_img', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('out_img', 1280, 720)
        # cv2.imshow("out_img", out)
        # cv2.imwrite('./test.jpg', out)
        # TODO 保存视频
        # output_video.write(out)

    rtscap.stop_read()
    rtscap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # ap = argparse.ArgumentParser(description="Demo of argparse")
    # ap.add_argument('--model', default='Video')
    # ap.add_argument("-i",
    #                 "--input",
    #                 default="./input_video/",
    #                 help="path to input video")
    # ap.add_argument("-o",
    #                 "--output_path",
    #                 default="./output_video/",
    #                 help="path to output video")
    # ap.add_argument("-r",
    #                 "--rtsp_url",
    #                 default="rtsp://localhost:8554/mystream")
    # args = ap.parse_args()
    args = parse_args()
    model = load_model(args.model, args.model_type, args.backend)
    process_video(model=model,rtsp_url=args.rtsp_url, output_path=args.output_path)
    if args.model == "Image":
        pass
