'''
Author: your name
Date: 2021-07-23 16:45:41
LastEditTime: 2021-08-16 15:20:40
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /lanenet-lane-detection-pytorch/Volumes/Public/CRRC/CRRC_Data/2021-07-28 摄像头数据/车内/晚上/videos2pictures.py
'''
import os
import cv2
import glob

videos_src_path = '../傍晚/'  # 视频文件夹路径
videos_save_path = '../傍晚/frames'  # 要输出的图片路径
# videos = os.listdir(videos_src_path)
videos = os.glob.glob(os.path.join(videos_src_path,"*.mp4")
# videos.sort(key=lambda x:int(x[5:-4]))
i = 1
for each_video in videos:
    if not os.path.exists(videos_save_path):
        os.mkdir(videos_save_path)
    each_video_save_full_path = videos_save_path + '/' + str(i) + '_'
    each_video_full_path = os.path.join(videos_src_path, each_video)
    cap = cv2.VideoCapture(each_video_full_path)
    frame_count = 1
    frame_clip = 6  # 采样的间隔
    success = True
    while (success):
        success, frame = cap.read()
        if success:
            if frame_count % frame_clip == 1:
                if frame.shape[1] != 704:
                    frame = cv2.resize(frame, (704, 576))
                cv2.imwrite(
                    each_video_save_full_path + "frame%d.jpg" % frame_count,
                    frame)
                print('正在写入%d张图片' % frame_count)
            frame_count = frame_count + 1
        # success = False
    i = i + 1
    cap.release()
