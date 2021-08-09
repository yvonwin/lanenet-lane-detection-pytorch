'''
Author: your name
Date: 2021-08-03 14:14:42
LastEditTime: 2021-08-05 16:07:36
LastEditors: Please set LastEditors
Description: 批量生成tusimple格式
FilePath: /labelme处理/convert1.py
'''

import cv2
from skimage import measure, color
from skimage.measure import regionprops
import numpy as np
import os
import copy


def replace_color(img, src_clr, dst_clr):
    ''' 通过矩阵操作颜色替换程序
    @param  img:    图像矩阵
    @param  src_clr:    需要替换的颜色(r,g,b)
    @param  dst_clr:    目标颜色        (r,g,b)
    @return             替换后的图像矩阵
    '''
    img_arr = np.asarray(img, dtype=np.double)

    r_img = img_arr[:, :, 0].copy()
    g_img = img_arr[:, :, 1].copy()
    b_img = img_arr[:, :, 2].copy()

    img = r_img * 256 * 256 + g_img * 256 + b_img
    src_color = src_clr[0] * 256 * 256 + src_clr[1] * 256 + src_clr[2]  #编码

    r_img[img == src_color] = dst_clr[0]
    g_img[img == src_color] = dst_clr[1]
    b_img[img == src_color] = dst_clr[2]

    dst_img = np.array([r_img, g_img, b_img], dtype=np.uint8)
    dst_img = dst_img.transpose(1, 2, 0)

    return dst_img


def skimageFilter(gray):  #根据标签颜色，标准化标签如:

    binary_warped = copy.copy(gray)
    binary_warped[
        binary_warped > 0.1] = 255  # 注释后取消resize生成的binary_image有问题 为全黑
    binary_warped = replace_color(binary_warped, (128, 0, 0), (1, 1, 1))  #红色是1
    binary_warped = replace_color(binary_warped, (0, 128, 0), (2, 2, 2))  #绿色是2
    binary_warped = replace_color(binary_warped, (0, 128, 128),
                                  (3, 3, 3))  #黄色是3
    binary_warped = replace_color(binary_warped, (0, 0, 128), (4, 4, 4))  #蓝色是4

    return binary_warped


def skimageFilter2(gray):

    binary_warped = copy.copy(gray)
    binary_warped[binary_warped > 0.1] = 255

    gray = (np.dstack((gray, gray, gray)) * 255).astype('uint8')
    labels = measure.label(gray[:, :, 0], connectivity=1)
    dst = color.label2rgb(labels, bg_label=0, bg_color=(0, 0, 0))
    gray = cv2.cvtColor(np.uint8(dst * 255), cv2.COLOR_RGB2GRAY)
    return binary_warped, gray


def moveImageTodir(path, targetPath, name):
    if os.path.isdir(path):
        gt_image_dir = os.path.join(targetPath, 'gt_image')
        gt_binary_dir = os.path.join(targetPath, 'gt_binary_image')
        gt_instance_dir = os.path.join(targetPath, 'gt_instance_image')

        os.makedirs(gt_image_dir, exist_ok=True)
        os.makedirs(gt_binary_dir, exist_ok=True)
        os.makedirs(gt_instance_dir, exist_ok=True)
        image_name =  "gt_image/" + str(name) + ".png"  #原图
        binary_name = "gt_binary_image/" + str(name) + ".png"  #标签图
        instance_name = "gt_instance_image/" + str(name) + ".png"
        

        train_rows = targetPath+'/'+image_name + " " +targetPath+'/' +binary_name + " " +targetPath+'/'+instance_name + "\n"

        #train_rows = image_name + " " + binary_name + " " + "1 1 1 1" + "\n"  #数据标注内容，可自定义

        origin_img = cv2.imread(path + "/img.png")
        #origin_img = cv2.resize(origin_img, (704, 576))
        cv2.imwrite(targetPath + "/" + image_name, origin_img)

        img = cv2.imread(path + '/label.png')
        #img = cv2.resize(img, (704, 576))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_warped, instance = skimageFilter2(gray)
        #binary_warped = cv2.cvtColor(binary_warped, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(targetPath + "/" + binary_name, binary_warped)
        cv2.imwrite(targetPath + "/" + instance_name, instance)
        print("success create data name is : ", train_rows)
        return train_rows
    return None


if __name__ == "__main__":
    #ep = "D:/File/Winscp/20.png"
    #img = cv2.imread(ep)
    #getimagelabel(img)
    count = 1
    with open("./train.txt", 'w+') as file:

        dir_name = "./test_dir/01/"  #os.path.join("./images", images_dir + "/annotations")#上一个生成的文件目录，即保存转换好的标签的目录
        for annotations_dir in os.listdir(dir_name):
            #print("********", annotations_dir)
            json_dir = os.path.join(dir_name, annotations_dir)
            if os.path.isdir(json_dir):

                train_rows = moveImageTodir(json_dir, "/Users/wk/Desktop/labelme处理/test_dir/test",
                                            str(count).zfill(4))
                file.write(train_rows)
                count += 1