'''
Author: your name
Date: 2021-08-05 17:41:49
LastEditTime: 2021-08-05 18:40:14
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /lanenet-lane-detection-pytorch/test_lanenet.py
'''
import  argparse
import  os
import  os.path as ops
import time

import cv2
import torch
from model.lanenet.LaneNet import LaneNet
from torchvision import  transforms
from model.utils.cli_helper_test import  parse_args
from model.utils.postprocess import  embedding_post_process
import  numpy as np
from PIL import  Image
import  glob

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_args():
    """
    :return
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, default='./log/best_model.pth', help='The path of weights to be tested')
    parser.add_argument('--src_dir', type=str, default='./data/test_img/', help='The path of image to be tested')
    parser.add_argument('--save_dir', type=str, default='./test_output', help='The path of save output')
    return parser.parse_args()    


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
    image_list = glob.glob('{:s}/**/*.jpg'.format(args.src_dir), recursive=True)
    print(image_list)

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

    for img_path in image_list:
        img_name = img_path.split('/')[-1]
        dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
        dummy_input = torch.unsqueeze(dummy_input, dim=0)
        outputs = model(dummy_input)

        input = Image.open(img_path)
        input = input.resize((resize_width, resize_height))
        input = np.array(input)

        # TODO output压到一维聚类
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
        cv2.imwrite(os.path.join(save_dir, 'result_' + img_name), img)

        cv2.imwrite(os.path.join(save_dir, 'instance_output' + img_name),
                    instance_pred.transpose((1, 2, 0)))
        cv2.imwrite(os.path.join(save_dir, 'binary_output' + img_name), binary_pred)



if __name__ == '__main__':
    test_lanenet()