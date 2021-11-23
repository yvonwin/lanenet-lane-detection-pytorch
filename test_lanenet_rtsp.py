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
from PIL import Image, ImageDraw

"""读取rtsp流并处理图片
读取相机rtsp流，并且处理作为socket通信服务器，处理图片。
Todo:
标准化： 考虑异常情况
异常情况汇总:
1. 未检测到rtsp流 增加了报错，但是要全部退出程序，还需要close掉server端 已解决 目前策略，一开始没检测到rtsp流，就直接退出。 done. 11.23在工控机下测试发现bug.在虚拟机中开机rtsp流，但是还是get到异常，暂时改回去了。
2. 后续可能相机突然断流，也需要异常关闭. 这个比较容易实现 看需要的时候再实现
3. 异常关闭后自启. 非此程序内部逻辑，需要脚本检查控制： 目前初步思路是查看端口占用，如果端口占用，什么都不做，如果没占用，重启程序。
4. opencv需要3.4.0版本，安装4.0版本会报错，后处理阶段有个函数接口有变化。

功能完善：
1. 功能聚合。车道与障碍物检测融合.
2. 稳定性测试 


"""

FPS = 25
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
HOST, PORT = "127.0.0.1", 8000


status = 0
num_boxes = 0
rtsp_status = 1

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    """通信模块使用socketserver异步监听，每次接收到数据后，调用handle方法，绘制激光雷达数据。
    Args:
        socketserver ([type]): [description]
    """

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
            else:
                status = 0

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
    infer_start_time = time.time()
    outputs = model(dummy_input)
    print("infer time: ", time.time() - infer_start_time)
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
    cluster_start_time = time.time()
    mask_image, _, _, _ = cluster.get_lane_mask(
        instance_seg_ret=instance_pred, binary_seg_ret=binary_pred, gt_image=input
    )
    print("cluster处理时间为：", time.time() - cluster_start_time)
    # TODO get left_lines, right_lines
    

    print("**status ok!**",status)
    # draw object box. draw_boxes()
    draw_mask=cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR) 
    # 在无障碍物下展示
    cv2.imshow('draw_mask', draw_mask)
    if status:
        if num_boxes > 0:
            # TODO 筛选
            # 思路 根据聚类出来的点作为边界点进行判断。
            # 判断障碍物的边界点是否在车道线上。
            # 如果在车道线上，则认为是障碍物。 我打算用先拿box的顶点坐标取y 判断是否在x1和x2之间
            # 1. get box顶点坐标, 这一步有点麻烦，因为是一个点集，先简化问题，取顶点。
            # 2. 判断是否在x1和x2之间，如果不再 remove box。这一部分主要是车道的限界问题。
            # 3. 如果在，则认为是障碍物。
            if boxes is not None:
                print('进入画框选择')
                for box in boxes[:int(num_boxes)]:
                    # x1,x2 为左上角  y1 y2为右下角
                    #print(box)
                    # 原始的size
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[0]-box[2])
                    y2 = int(box[1]-box[3])
                    # 画原始帧，方便观测效果
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 0)
                    ##  放缩坐标
                    # resize box
                    x1 = x1/(704/512)
                    y1 = y1/(576/256)
                    x2 = x2/(704/512)
                    y2 = y2/(576/256)
                    # 画放缩帧，方便观测效果
                    cv2.rectangle(draw_mask, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 0)
                    # draw = ImageDraw.Draw(Image.fromarray(draw_image))
                    #draw.rectangle((x1,y1,x1,y2) ,outline=(255,0,0))
    #    cv2.imwrite('draw_frame.png', frame)
    #    cv2.imwrite('draw_mask.png',draw_mask)
        cv2.imshow('draw_mask', draw_mask)

    #print(instance_pred.shape)
    #for i in range(3):
    #    instance_pred[:, :, i] = minmax_scale(instance_pred[:, :, i])
    # embedding_image = np.array(instance_pred, np.uint8)

    # 拓展binary_pred通道 方便可视化
    # bin_image = np.expand_dims(binary_pred, axis=2)
    # bin_image = np.concatenate((bin_image, bin_image, bin_image), axis=-1)
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
    # out_all = np.vstack(
    #     [
    #         np.hstack(
    #             [cv2.cvtColor(input, cv2.COLOR_RGB2BGR) / 255.0, cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR) / 255.0]
    #         ),
    #         np.hstack([instance_pred * 255, bin_image * 255]),
    #     ]
    # )
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
    return draw_mask


def process_video(model, rtsp_url, output_path):
    rtscap = RTSCapture.create(rtsp_url)
    if rtscap:
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
            # time1
            start_time1 = time.time()
            out = test_lanenet_one_img(model, frame)
            print('test_lanenet_oneimg use time:', time.time() - start_time1)
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
    else:
        global rtsp_status
        rtsp_status = 0


if __name__ == "__main__":
    args = parse_args()
    # 加载模型
    model = load_model(args.model, args.model_type, args.backend)
 
    print("start listening")
   # server = socketserver.ThreadingTCPServer((HOST, PORT), MyTCPHandler)
    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    #ip, port = server.server_address

    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    try:
        server_thread = threading.Thread(target=server.serve_forever)
        # # # Exit the server thread when the main thread terminates
        server_thread.daemon = False
        server_thread.start()
        print("Server loop running in thread:", server_thread.name)
    # ctrlc 退出时关闭线程
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()
    # 循环处理接收的图片
    process_video(model=model, rtsp_url=args.rtsp_url, output_path=args.output_path)
    if not rtsp_status:
        server.shutdown()
        server.server_close()
        print("server closed")
    # if args.model == "Image":
    #     pass
