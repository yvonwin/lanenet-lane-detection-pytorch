'''
Author: your name
Date: 2021-11-23 18:03:03
LastEditTime: 2021-12-06 15:19:46
LastEditors: your name
Description: 文件批量重命名
FilePath: /lanenet-lane-detection-pytorch/local_utils/rename.py
'''
import os
path_name='./images'
#path_name :表示你需要批量改的文件夹
i=0
for item in os.listdir(path_name):#进入到文件夹内，对每个文件进行循环遍历
    os.rename(os.path.join(path_name,item),os.path.join(path_name,(str(i)+'.jpg')))#os.path.join(path_name,item)表示找到每个文件的绝对路径并进行拼接操作
    i+=1
