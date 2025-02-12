"""
Author: your name
Date: 2021-08-23 15:04:30
LastEditTime: 2021-08-26 10:26:38
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /lanenet-lane-detection-pytorch/model/utils/lanenet_postprocess.py
"""

"""
LaneNet模型后处理
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2


class LaneNetPoseProcessor(object):
    """ """

    def __init__(self):
        """ """
        pass

    @staticmethod
    def _morphological_process(image, kernel_size=5):
        """

        :param image:
        :param kernel_size:
        :return:
        """
        if image.dtype is not np.uint8:
            image = np.array(image, np.uint8)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

        # close operation fille hole
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

        return closing

    @staticmethod
    def _connect_components_analysis(image):
        """

        :param image:
        :return:
        """
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)

    def postprocess(self, image, minarea_threshold=15):
        """

        :param image:
        :param minarea_threshold: 连通域分析阈值
        :return:
        """
        # 首先进行图像形态学运算
        morphological_ret = self._morphological_process(image, kernel_size=5)

        # 进行连通域分析
        connect_components_analysis_ret = self._connect_components_analysis(image=morphological_ret)

        # 排序连通域并删除过小的连通域
        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]

        for index, stat in enumerate(stats):
            if stat[4] <= minarea_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        return morphological_ret


def main():
    processor = LaneNetPoseProcessor()
    image = cv2.imread(
        "/Users/wk/Desktop/Mac_Workspaces/lanenet-lane-detection-pytorch/test_output1/binary_output0071.png",
        cv2.IMREAD_GRAYSCALE,
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    postprocess_ret = processor.postprocess(image)

    plt.figure("src")
    plt.imshow(image)
    plt.figure("post")
    plt.imshow(postprocess_ret)
    plt.show()


if __name__ == "__main__":
    main()
