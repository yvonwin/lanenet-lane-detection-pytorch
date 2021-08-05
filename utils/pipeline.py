'''
Author: your name
Date: 2021-08-02 18:14:51
LastEditTime: 2021-08-04 10:17:58
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /labelme处理/pipeline.py
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import  glob
 
if __name__ == '__main__':
    # for scan_dir in ['test_dir/01']
    for scan_dir in glob.glob('test_dir/*'):#这里写自己的图片的路径
        json_data = scan_dir
        print(json_data)
        for name in os.listdir(json_data):
            file_path = os.path.join(json_data, name)
            print(file_path)
            if os.path.isfile(file_path) and name.split('.')[1] == 'json': # 判断是json文件
                os.system(str("labelme_json_to_dataset " + file_path))
                print("success json to dataset: ", file_path)