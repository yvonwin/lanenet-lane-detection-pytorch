#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-15 下午4:29
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_cluster.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet中实例分割的聚类部分 暂未使用
"""
import numpy as np
import glog as log
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import time
import warnings
import cv2
import matplotlib.pyplot as plt


class LaneNetCluster(object):
    """
    实例分割聚类器
    """

    def __init__(self):
        """ """
        self._color_map = [
            np.array([255, 0, 0]),
            np.array([0, 255, 0]),
            np.array([0, 0, 255]),
            np.array([125, 125, 0]),
            np.array([0, 125, 125]),
            np.array([125, 0, 125]),
            np.array([50, 100, 50]),
            np.array([100, 50, 100]),
        ]
        pass

    @staticmethod
    def _cluster(prediction, bandwidth):
        """
        实现论文SectionⅡ的cluster部分
        :param prediction:
        :param bandwidth:
        :return:
        """
        ms = MeanShift(bandwidth, bin_seeding=True)
        log.info("开始Mean shift聚类 ...")
        tic = time.time()
        try:
            ms.fit(prediction)
        except ValueError as err:
            log.error(err)
            return 0, [], []
        log.info("Mean Shift耗时: {:.5f}s".format(time.time() - tic))
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        num_clusters = cluster_centers.shape[0]

        log.info("聚类簇个数为: {:d}".format(num_clusters))

        return num_clusters, labels, cluster_centers

    @staticmethod
    def _cluster_v2(prediction):
        """
        dbscan cluster
        :param prediction:
        :return:
        """
        # db = DBSCAN(eps=0.7, min_samples=200).fit(prediction)
        # log.info("开始dbscan聚类")
        tic = time.time()
        # 由于存在细线问题（标注数据导致）设置min_Samples为20
        # db = DBSCAN(eps=0.35, min_samples=20)
        db = DBSCAN(eps=0.35, min_samples=200)
        try:
            # 去除均值并缩放到单位方差来标准化特征
            features = StandardScaler().fit_transform(prediction)
            db.fit(features)
            # db.fit(prediction)
        except Exception as err:
            log.error(err)
            log.info("这张图片异常，可能是曝光过度？")
            return 0, [], []
        if db:
            db_labels = db.labels_
            unique_labels = np.unique(db_labels)
            unique_labels = [tmp for tmp in unique_labels if tmp != -1]
            # log.info("聚类簇个数为: {:d}".format(len(unique_labels)))

            num_clusters = len(unique_labels)
            cluster_centers = db.components_

            # log.info("dbscan耗时: {:.5f}s".format(time.time() - tic))

            return num_clusters, db_labels, cluster_centers

    @staticmethod
    def _get_lane_area(binary_seg_ret, instance_seg_ret):
        """
        通过二值分割掩码图在实例分割图上获取所有车道线的特征向量
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        # print(binary_seg_ret.shape, instance_seg_ret.shape)
        idx = np.where(binary_seg_ret == 1)

        lane_embedding_feats = []
        lane_coordinate = []
        for i in range(len(idx[0])):
            lane_embedding_feats.append(instance_seg_ret[idx[0][i], idx[1][i]])
            lane_coordinate.append([idx[0][i], idx[1][i]])
        print(len(idx[0]))
        return np.array(lane_embedding_feats, np.float32), np.array(lane_coordinate, np.int64)

    @staticmethod
    def _thresh_coord(coord):
        """
        过滤实例车道线位置坐标点,假设车道线是连续的, 因此车道线点的坐标变换应该是平滑变化的不应该出现跳变
        :param coord: [(x, y)]
        :return:
        """
        pts_x = coord[:, 0]
        mean_x = np.mean(pts_x)

        idx = np.where(np.abs(pts_x - mean_x) < mean_x)

        return coord[idx[0]]

    @staticmethod
    def _lane_fit(lane_pts):
        """
        车道线多项式拟合
        :param lane_pts:
        :return:
        """
        if not isinstance(lane_pts, np.ndarray):
            lane_pts = np.array(lane_pts, np.float32)

        x = lane_pts[:, 0]
        y = lane_pts[:, 1]
        x_fit = []
        y_fit = []
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                f1 = np.polyfit(y, x, 3)
                p1 = np.poly1d(f1)
                y_min = int(np.min(y))
                y_max = int(np.max(y))
                y_fit = []
                for i in range(y_min, y_max + 1):
                    y_fit.append(i)
                x_fit = p1(y_fit)
            except Warning as e:
                log.error(e)
                x_fit = x
                y_fit = y
            finally:
                return zip(x_fit, y_fit)

    def get_lane_mask(self, binary_seg_ret, instance_seg_ret, gt_image):
        """

        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        lane_embedding_feats, lane_coordinate = self._get_lane_area(binary_seg_ret, instance_seg_ret)
        # 使用meanshift
        # num_clusters, labels, cluster_centers = self._cluster(lane_embedding_feats, bandwidth=4.)
        # 使用dbscan
        num_clusters, labels, cluster_centers = self._cluster_v2(lane_embedding_feats)

        # 聚类簇超过八个则选择其中类内样本最多的八个聚类簇保留下来 这里我改成两个。
        if num_clusters > 2:
            cluster_sample_nums = []
            for i in range(num_clusters):
                cluster_sample_nums.append(len(np.where(labels == i)[0]))
            sort_idx = np.argsort(-np.array(cluster_sample_nums, np.int64))
            cluster_index = np.array(range(num_clusters))[sort_idx[0:2]]
        else:
            cluster_index = range(num_clusters)

        mask_image = np.copy(gt_image)
        for index, i in enumerate(cluster_index):
            if i == -1:
                continue
            idx = np.where(labels == i)
            # print(idx)
            # print(len(idx[0]))
            coord = lane_coordinate[idx]
            # print(coord)
            # color = (int(self._color_map[index][0]),
            #         int(self._color_map[index][1]),
            #         int(self._color_map[index][2]))
            color = (0, 255, 0)
            coord = np.array(coord)
            #print(coord.shape)
            mask_image[coord[:, 0], coord[:, 1], :] = color
        return mask_image, lane_coordinate, cluster_index, labels


if __name__ == "__main__":
    binary_seg_image = cv2.imread(
        "/Users/wk/Desktop/Mac_Workspaces/lanenet-lane-detection-pytorch/test_output_1/binary_output0298.png",
        cv2.IMREAD_GRAYSCALE,
    )
    binary_seg_image[np.where(binary_seg_image == 255)] = 1
    instance_seg_image = cv2.imread(
        "/Users/wk/Desktop/Mac_Workspaces/lanenet-lane-detection-pytorch/test_output_1/instance_output0298.png",
        cv2.IMREAD_UNCHANGED,
    )
    gt_image = cv2.imread(
        "/Users/wk/Desktop/Mac_Workspaces/lanenet-lane-detection-pytorch/test_output_1/input_0298.png",
        cv2.IMREAD_UNCHANGED,
    )
    ele_mex = np.max(instance_seg_image, axis=(0, 1))
    for i in range(3):
        if ele_mex[i] == 0:
            scale = 1
        else:
            scale = 255 / ele_mex[i]
        instance_seg_image[:, :, i] *= int(scale)
    embedding_image = np.array(instance_seg_image, np.uint8)
    cluster = LaneNetCluster()
    t_start = time.time()
    mask_image = cluster.get_lane_mask(
        instance_seg_ret=instance_seg_image, binary_seg_ret=binary_seg_image, gt_image=gt_image
    )
    t_cost = time.time() - t_start
    print("单张图像车道线聚类耗时: {:.5f}s".format(t_cost))

    plt.figure("mask_iamge")
    plt.imshow(mask_image[0][:, :, (2, 1, 0)])
    plt.figure("instance_image")
    plt.imshow(embedding_image[:, :, (2, 1, 0)])
    plt.show()

    # cv2.imwrite('./mask_image.png', mask_image[0])
