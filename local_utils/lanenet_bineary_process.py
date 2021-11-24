import cv2
import numpy as np


# 求最大连通域的中心点坐标
def centroid(max_contour):
    # 得到图像的距
    moment = cv2.moments(max_contour)
    if moment["m00"] != 0:
        # 计算重心
        cx = int(moment["m10"] / moment["m00"])
        cy = int(moment["m01"] / moment["m00"])
        return cx, cy
    else:
        return 0, 0


def get_select_bineary(curimg, selcontour):
    """计算二进制图中两条轨道线区域内的有效轨道轮廓
    :param selcontour:二进制中检测的两条轨道轮廓：存在两条轨道线区域相互链接，岔道干扰的情况
    :return:过滤费轨道去线后的轨道线区域的轮廓
    """
    # 获取最大的连通区域内的坐标值确定轨道区域的大致范围过滤误差干扰
    sort_selcontour = []
    if cv2.contourArea(selcontour[0]) >= cv2.contourArea(selcontour[1]):
        sort_selcontour.append(selcontour[0])
        sort_selcontour.append(selcontour[1])
    else:
        sort_selcontour.append(selcontour[1])
        sort_selcontour.append(selcontour[0])

    # 计算最大轮廓的坐标点信息确定轨道区域的坐标点信息
    x_cntcenter0, y_cntcenter0 = centroid(sort_selcontour[0])  # 最大值的中心轮廓坐标
    x_cntcenter1, y_cntcenter1 = centroid(sort_selcontour[1])  # 另外一个轮廓的中心坐标值
    sort0_selcontour = []
    sort1_selcontour = []

    if abs(x_cntcenter1 - x_cntcenter0) < 80:  # 轨道宽度最大为80个像素   过滤误差点的干扰
        mask = curimg
        # 去掉岔道场景中的另外两条轨道的干扰
        sort0_selcontour = sorted(sort_selcontour[0], key=lambda x: (x[:][0][1], x[:][0][0]))  # y值升序排列，x值升序排列
        # sort1_selcontour = sorted(sort_selcontour[1], key=lambda x: (-x[:][0][1],-x[:][0][0]))  # y值降序排列，x值升序排列

        # 判断最大区域是否为2两轨道线重合轮廓，重合则直接擦去第二条，否则则保留2条：
        # 取y值最小的点作为远端的起点，沿y轴增大的方向水平绘制直线，确认直线与轮廓的交点是否满足条件：
        # 交点x的差距大于10个像素点则为两条直线轨道重合轮廓 否则即为2条分开的轨道轮廓
        xdelta_railline = []  # 单个轮廓的x的最小值点,最大值点
        num0 = 0
        curleftY_front = 0
        curleftY = 0
        curleftx = 0
        while num0 < len(sort0_selcontour):
            curleftY_front = curleftY
            curleftY = sort0_selcontour[num0][0][1]  # 当前y值
            if curleftY == curleftY_front:
                break

            if num0 >= 1:
                xdelta_railline.append(sort0_selcontour[num0 - 1][0][0] - curleftx)  # 单轨道轮廓的x差值
                curleftx = sort0_selcontour[num0][0][0]  # 左侧轨道装载最小值X
            else:
                curleftx = sort0_selcontour[num0][0][0]

            for num1 in range(num0 + 1, len(sort0_selcontour)):
                if curleftY == sort0_selcontour[num1][0][1]:
                    pass
                else:
                    break
            num0 = num1

        # 根据单条轮廓最大值与最小值的差约为10个像素值，而2条轨道相交为一条轮廓情况下左右边界远大于10 的情况进行判断是为单轨还是双规
        num_cnt = len([cnt for cnt in xdelta_railline if cnt > 20])  # 统计轨道的x差异
        print(num_cnt)
        if num_cnt > 3:
            mask = cv2.drawContours(curimg, sort_selcontour, 1, 0, cv2.FILLED)  # 填充不需要的轮廓点
    else:
        mask = cv2.drawContours(curimg, sort_selcontour, 1, 0, cv2.FILLED)  # 填充不需要的轮廓点
    return mask


def get_binearycontour(binaryimg):
    """
    :param binaryimg:二进制图像bineary图像
    :return:图中联通区域的数量，联通区域的轮廓
    """
    contours = []
    binaryimg = binaryimg.astype(np.uint8)
    # opecv3
    _, contours, hierarchy = cv2.findContours(binaryimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # opencv4
    # contours, hierarchy = cv2.findContours(binaryimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    b_contournum = len(contours)
    # print("number of contours:%d" % b_contournum)
    if b_contournum == 0:  # 不存在连通区域则直接返回数据为0
        return binaryimg

    # 找到最大的两个联通区域,其余的联通区域填充为0
    selcontour = []
    if b_contournum == 1:
        mask = binaryimg
        final_contour = contours
    elif b_contournum == 2:
        for i in range(b_contournum):
            selcontour.append(contours[i])
        mask = get_select_bineary(binaryimg, selcontour)
    else:  # 存在误差数据情况，则取最大的两个联通区域
        index_contour = []
        for i in range(b_contournum):
            index_contour.append((i, cv2.contourArea(contours[i])))
        sortindex_len = sorted(index_contour, key=lambda x: -x[1])  # 面积降序排列
        selcontour.append(contours[sortindex_len[0][0]])  # 保存两条最长的轮廓
        selcontour.append(contours[sortindex_len[1][0]])
        # 将超过2个联通区域的图填充为黑色
        for j in range(2, b_contournum):
            curimg = cv2.drawContours(binaryimg, contours, sortindex_len[j][0], 0, cv2.FILLED)  # 填充不需要的轮廓点
        # cv2.imwrite('./model/test_output/test01.jpg',curimg * 255)

        # 获取最大的连通区域内的坐标值确定轨道区域的大致范围过滤误差干扰
        mask = get_select_bineary(curimg, selcontour)
    return mask
