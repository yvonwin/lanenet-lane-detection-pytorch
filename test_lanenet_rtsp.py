"""
Author: yvon
Date: 2021-07-19 14:26:06
LastEditTime: 2021-08-26 17:06:11
LastEditors: Please set LastEditors
Description: 实时拉取rtsp测试
FilePath: /refact_7.13/pipeline.py
"""
import os
import cv2
import numpy as np

# import math
# import argparse
from RTSCapture import RTSCapture

# import os.path as ops
import time
# from sklearn import cluster
import torch
from model.lanenet.LaneNet import LaneNet
from torchvision import transforms
from model.utils.cli_helper_test import parse_args
# from local_utils.lanenet_data_process import lanenet_data_process
from local_utils.lanenet_bineary_process import get_binearycontour


# from model.utils.postprocess import embedding_post_process
from PIL import Image

# import glob
from model.utils import lanenet_cluster
# from model.utils import lanenet_postprocess
# import matplotlib.pyplot as plt

import socketserver
import struct
import threading
import time


FPS = 25
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

HOST, PORT = "127.0.0.1", 7000


status = 0
num_boxes = 0

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def setup(self):
        ip = self.client_address[0].strip()     # 获取客户端的ip
        port = self.client_address[1]           # 获取客户端的port
        print(ip+":"+str(port)+" is connect!")

    def handle(self):
        while True:
            self.data = self.request.recv(336).strip()
            # cur_thread = threading.current_thread()
            # print(cur_thread)
            # print(self.data)
            if self.data:
                news = struct.unpack('ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff', self.data)
                #print(news)
                # print(type(news))  # is tuple
                # 取出status，需在这一步定位全局变量，作为是否绘制障碍物的标志
                global num_boxes
                global boxes
                global status
                status  = news[0]
                if status:
                    num_boxes = news[1]
                    l = list(news[4:])
                    print('l长度为',len(l))
                    # 拆分boxes
                    n = 4
                    boxes=[l[i:i + n] for i in range(0, len(l), n)]
                    print("receive boxes成功, boxes为",boxes)
                    print('num_boxes为',num_boxes)
            # else:
            #     status = 0

    def finish(self):
        print('client is disconnected')

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass
        

# def load_test_data(img_path, transform):
#     img = Image.open(img_path)
#     img = transform(img)
#     return img


def load_model(model_path, model_type, backend):
    model = LaneNet(arch=model_type, backend=backend)
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)  # 默认保存的模型是gpu
    else:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))  # cpu推理
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

    """ "
    :param src_dir:
    :param weights_path:
    :return:
    """
    args = parse_args()
    resize_height = args.height
    resize_width = args.width
    save_dir = args.save

    os.makedirs(save_dir, exist_ok=True)
    data_transform = transforms.Compose(
        [
            transforms.Resize((resize_height, resize_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    input = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    dummy_input = data_transform(input)
    dummy_input = dummy_input.to(DEVICE)

    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    outputs = model(dummy_input)

    # input = Image.open(img_path)
    input = input.resize((resize_width, resize_height))
    input = np.array(input)

    instance_pred = torch.squeeze(outputs["instance_seg_logits"].detach().to("cpu")).numpy()
    binary_pred = torch.squeeze(outputs["binary_seg_pred"]).to("cpu").numpy()

    # 开始后处理
    instance_pred = instance_pred.transpose(1, 2, 0)
    cluster = lanenet_cluster.LaneNetCluster()
    # 删除一些比较小的联通区域
    # postprocessor = lanenet_postprocess.LaneNetPoseProcessor()
    # binary_pred = postprocessor.postprocess(binary_pred)
    # process
    start_time = time.time()
    binary_pred = get_binearycontour(binary_pred)  # 修改bineary图像处理使用
    print("binerycontour处理时间为：", time.time() - start_time)
    mask_image, _, _, _ = cluster.get_lane_mask(
        instance_seg_ret=instance_pred, binary_seg_ret=binary_pred, gt_image=input
    )
    # todo 筛选

    ## TODO draw line
    print("********",status)
    # draw object box. draw_boxes()
    if status:
        if num_boxes > 0:
            if boxes is not None:
                print('进入画框选择')
                for box in boxes[:int(num_boxes)]:
                    # x1,x2 为左上角  y1 y2为右下角
                    # draw box
                    print(box)
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0]-box[2]), int(box[1]-box[3])), (255, 0, 0), 2)
        cv2.imwrite('filename.png', frame)

    print(instance_pred.shape)
    for i in range(3):
        instance_pred[:, :, i] = minmax_scale(instance_pred[:, :, i])
    # embedding_image = np.array(instance_pred, np.uint8)

    # 拓展binary_pred通道 方便可视化
    bin_image = np.expand_dims(binary_pred, axis=2)
    bin_image = np.concatenate((bin_image, bin_image, bin_image), axis=-1)
    # 结果可视化 imwrite使用
    # out_all = np.vstack(
    #     [
    #         np.hstack(
    #             [cv2.cvtColor(input, cv2.COLOR_RGB2BGR), cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR)]
    #         ),
    #         np.hstack([instance_pred * 255, bin_image * 255]),
    #     ]
    # )
    # 修改归一化，便于imread
    out_all = np.vstack(
        [
            np.hstack(
                [cv2.cvtColor(input, cv2.COLOR_RGB2BGR) / 255.0, cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR) / 255.0]
            ),
            np.hstack([instance_pred * 255, bin_image * 255]),
        ]
    )
    # cv2.imshow("out_all", out_all)

    # cv2.imwrite(os.path.join(save_dir, 'input_' + img_name), input)
    # cv2.imwrite(os.path.join(save_dir, 'result_' + img_name), mask_image)

    # cv2.imwrite(os.path.join(save_dir, 'instance_output' + img_name),
    #            instance_pred * 255)
    # cv2.imwrite(os.path.join(save_dir, 'binary_output' + img_name),
    #            binary_pred * 255)
    # cv2.imwrite(os.path.join(save_dir, 'out_all.png'), out_all)
    # plt.figure('out_all')
    # plt.imshow(out_all[:, :, (2, 1, 0)])
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
    # out_videofile = output_path + '_test.mp4'
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # output_video = cv2.VideoWriter(out_videofile, fourcc, FPS,
    #                               (int(framewidth), int(frameheight)))
    print("start to write video")
    while rtscap.isStarted():
        ok, frame = rtscap.read_latest_frame()  # read_latest_frame() 替代 read() 此时取到的为最新的帧
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break
        if not ok:
            continue
        # 帧处理
        out = test_lanenet_one_img(model, frame)

        # print(out)
        # cv2.namedWindow('out_img', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('out_img', 1024, 756)

        # cv2.imshow("out_img", out)

        # cv2.namedWindow('out_img', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('out_img', 1280, 720)
        # cv2.imshow("out_img", out)
        cv2.imwrite('./test.jpg', out)
        # TODO 保存视频
        # output_video.write(out)

    rtscap.stop_read()
    rtscap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.model, args.model_type, args.backend)
 
    print("listening")
   # server = socketserver.ThreadingTCPServer((HOST, PORT), MyTCPHandler)
    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    #ip, port = server.server_address

    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread = threading.Thread(target=server.serve_forever)
    # # # Exit the server thread when the main thread terminates
    server_thread.daemon = False
    server_thread.start()
    print("Server loop running in thread:", server_thread.name)
    process_video(model=model, rtsp_url=args.rtsp_url, output_path=args.output_path)
    if args.model == "Image":
        pass
