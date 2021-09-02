"""
Author: yvon
Date: 2021-08-05 17:41:49
LastEditTime: 2021-08-26 19:31:10
LastEditors: Please set LastEditors
Description: 批量测试文件夹中的图片
FilePath: /lanenet-lane-detection-pytorch/test_lanenet.py
"""
# import argparse
import os
import os.path as ops

# import time
import cv2

# from sklearn import cluster
import torch
from model.lanenet.LaneNet import LaneNet
from torchvision import transforms
from model.utils.cli_helper_test import parse_args

# from model.utils.postprocess import embedding_post_process
import numpy as np
from PIL import Image
import glob
from model.utils import lanenet_cluster

# from model.utils import lanenet_postprocess
from local_utils.lanenet_bineary_process import get_binearycontour

import matplotlib.pyplot as plt
import glog as log
import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_test_data(img_path, transform):
    img = Image.open(img_path)
    img = transform(img)
    return img


def minmax_scale(input_arr):
    """
    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet_batch():
    """ "
    :param src_dir:
    :param weights_path:
    :return:
    """
    args = parse_args()
    resize_height = args.height
    resize_width = args.width
    save_dir = args.save
    assert ops.exists(args.src_dir), "{:s} not exist".format(args.src_dir)
    os.makedirs(save_dir, exist_ok=True)
    types = (".jpg", ".png", ".bmp", ".jpeg", ".JPG", ".JPEG")
    image_list = []
    for img_type in types:
        image_list.extend(glob.glob(os.path.join(args.src_dir, "**/*" + img_type), recursive=True))
    print(image_list)

    data_transform = transforms.Compose(
        [
            transforms.Resize((resize_height, resize_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # 注册加载模型
    model_path = args.model
    model = LaneNet(arch=args.model_type, backend=args.backend)
    model_load_start = time.time()
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)  # 默认保存的模型是gpu
    else:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))  # cpu推理
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    log.info("load model use time:{:.4f}".format(time.time() - model_load_start))

    for img_path in image_list:
        img_name = img_path.split("/")[-1]
        dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
        dummy_input = torch.unsqueeze(dummy_input, dim=0)
        infer_start_time = time.time()
        outputs = model(dummy_input)
        log.info(" infer time: {:.4f}s".format(time.time() - infer_start_time))
        input = Image.open(img_path)
        input = input.resize((resize_width, resize_height))
        input = np.array(input)

        instance_pred = torch.squeeze(outputs["instance_seg_logits"].detach().to("cpu")).numpy()
        binary_pred = torch.squeeze(outputs["binary_seg_pred"]).to("cpu").numpy()

        post_time_start = time.time()
        # 开始后处理
        instance_pred = instance_pred.transpose(1, 2, 0)
        cluster = lanenet_cluster.LaneNetCluster()
        # 删除一些比较小的联通区域
        # postprocessor = lanenet_postprocess.LaneNetPoseProcessor()
        # binary_pred = postprocessor.postprocess(binary_pred)
        # binary后处理
        binary_pred = get_binearycontour(binary_pred)
        log.info("post time: {:.4f}s".format(time.time() - post_time_start))
        log.info("*****img_name is: %s" % img_name)
        cluster_start_time = time.time()
        mask_image, _, _, _ = cluster.get_lane_mask(
            instance_seg_ret=instance_pred, binary_seg_ret=binary_pred, gt_image=input
        )
        log.info("cluster time:{:.4f}".format(time.time() - cluster_start_time))
        # TODO 发送聚类处理结果到c++

        # TODO 接收激光雷达点集结果

        # for i in range(3):
        #     instance_pred[:, :, i] = minmax_scale(instance_pred[:, :, i])
        # embedding_image = np.array(instance_pred, np.uint8)
        # 拓展binary_pred通道 方便可视化
        bin_image = np.expand_dims(binary_pred, axis=2)
        bin_image = np.concatenate((bin_image, bin_image, bin_image), axis=-1)
        # 结果可视化
        # out_all = np.vstack(
        #     [
        #         np.hstack(
        #             [
        #                 cv2.cvtColor(input, cv2.COLOR_RGB2BGR),
        #                 cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR),
        #             ]
        #         ),
        #         np.hstack([instance_pred * 255, bin_image * 255]),
        #     ]
        # )
        # 修改归一化，便于imread
        out_all = np.vstack(
            [
                np.hstack(
                    [
                        cv2.cvtColor(input, cv2.COLOR_RGB2BGR) / 255.0,
                        cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR) / 255.0,
                    ]
                ),
                np.hstack([instance_pred * 255, bin_image * 255]),
            ]
        )
        cv2.imshow("out_all", out_all)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        # cv2.imwrite(os.path.join(save_dir, 'input_' + img_name), input)
        # cv2.imwrite(os.path.join(save_dir, 'result_' + img_name), mask_image)

        # cv2.imwrite(os.path.join(save_dir, 'instance_output' + img_name),
        #            instance_pred * 255)
        # cv2.imwrite(os.path.join(save_dir, 'binary_output' + img_name),
        #            binary_pred * 255)
        # cv2.imwrite(os.path.join(save_dir, "out_all" + img_name), out_all)
        # plt.figure('out_all')
        # plt.imshow(out_all)
        # plt.imshow(mask_image[:, :, (2, 1, 0)])
        # plt.figure('embedding')
        # plt.imshow(embedding_image)
        # plt.show()


if __name__ == "__main__":
    test_lanenet_batch()
