'''
Author: your name
Date: 2021-08-05 13:52:48
LastEditTime: 2021-08-17 17:27:57
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /lanenet-lane-detection-pytorch/postprocess.py
'''
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import cv2


def embedding_post_process(embedding, bin_seg, band_width=1.5, max_num_lane=4):
    """
    First use mean shift to find dense cluster center.
    Arguments:
    ----------
    embedding: numpy [H, W, embed_dim]
    bin_seg: numpy [H, W], each pixel is 0 or 1, 0 for background pixel
    delta_v: coordinates within distance of 2*delta_v to cluster center are
    Return:
    ---------
    cluster_result: numpy [H, W], index of different lanes on each pixel
    """
    cluster_result = np.zeros(bin_seg.shape, dtype=np.int32)

    cluster_list = embedding[bin_seg > 0]
    if len(cluster_list) == 0:
        return cluster_result

    mean_shift = MeanShift(bandwidth=1.5, bin_seeding=True, n_jobs=-1)
    mean_shift.fit(cluster_list)

    labels = mean_shift.labels_
    cluster_result[bin_seg > 0] = labels + 1

    cluster_result[cluster_result > max_num_lane] = 0
    for idx in np.unique(cluster_result):
        if len(cluster_result[cluster_result == idx]) < 15:
            cluster_result[cluster_result == idx] = 0
    return cluster_result


def main(instance_pred, binary_pred):
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
    cv2.imwrite('./test_output/demo_result.png', img)
