'''
Author: your name
Date: 2021-08-05 17:41:49
LastEditTime: 2021-08-19 09:49:11
LastEditors: Please set LastEditors
Description: 批量测试文件夹中的图片
FilePath: /lanenet-lane-detection-pytorch/test_lanenet.py
'''
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
from model.utils import lanenet_cluster, lanenet_postprocess

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_test_data(img_path, transform):
    img = Image.open(img_path)
    img = transform(img)
    return img


def test_lanenet():
    """"
    :param src_dir:
    :param weights_path:
    :return:
    """

    args = parse_args()
    resize_height = args.height
    resize_width = args.width
    save_dir = args.save
    assert ops.exists(args.src_dir), '{:s} not exist'.format(args.src_dir)
    os.makedirs(save_dir, exist_ok=True)
    types = ('.jpg', '.png', '.bmp', '.jpeg', '.JPG', '.JPEG')
    image_list = []
    for img_type in types:
        image_list.extend(
            glob.glob(os.path.join(args.src_dir, '**/*' + img_type),
                      recursive=True))
    print(image_list)

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

    for img_path in image_list:
        img_name = img_path.split('/')[-1]
        dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
        dummy_input = torch.unsqueeze(dummy_input, dim=0)
        outputs = model(dummy_input)

        input = Image.open(img_path)
        input = input.resize((resize_width, resize_height))
        input = np.array(input)

        instance_pred = torch.squeeze(
            outputs['instance_seg_logits'].detach().to('cpu')).numpy()
        binary_pred = torch.squeeze(
            outputs['binary_seg_pred']).to('cpu').numpy()

        # postprocess
        instance_pred = instance_pred.transpose(1, 2, 0)
        cluster = lanenet_cluster.LaneNetCluster()
        # 删除一些比较小的联通区域
        postprocessor = lanenet_postprocess.LaneNetPoseProcessor()
        binary_pred = postprocessor.postprocess(binary_pred)
        print('*****fuck! img_name is: ', img_name)
        # TODO  曝光增强过滤
        mask_image, _, _, _ = cluster.get_lane_mask(
            instance_seg_ret=instance_pred,
            binary_seg_ret=binary_pred,
            gt_image=input)

        # 写结果
        image = np.expand_dims(binary_pred, axis=2)
        image = np.concatenate((image, image, image), axis=-1)

        out_all = np.vstack([
            np.hstack([
                cv2.cvtColor(input, cv2.COLOR_RGB2BGR),
                mask_image,
            ]),
            np.hstack([instance_pred * 255, image * 255])
        ])
        cv2.imwrite(os.path.join(save_dir, 'input_' + img_name), input)
        cv2.imwrite(os.path.join(save_dir, 'result_' + img_name), mask_image)

        cv2.imwrite(os.path.join(save_dir, 'instance_output' + img_name),
                    instance_pred * 255)
        cv2.imwrite(os.path.join(save_dir, 'binary_output' + img_name),
                    binary_pred * 255)
        cv2.imwrite(os.path.join(save_dir, 'out_all' + img_name), out_all)


if __name__ == '__main__':
    test_lanenet()
