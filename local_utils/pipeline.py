'''
Author: yvon
Date: 2021-08-02 18:14:51
LastEditTime: 2021-12-06 15:19:19
LastEditors: Please set LastEditors
Description: 批量处理labelme_json文件
FilePath: /labelme处理/pipeline.py
'''
import os
import glob
import argparse


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', type=str, help='json dir')
    return parser.parse_args()


def main(json_dir):
    if not os.path.exists(json_dir):
        raise Exception("{} not exists".format(json_dir))
    # 判断是否有子文件夹
    sub_dirs = [
        x for x in os.listdir(json_dir)
        if os.path.isdir(os.path.join(json_dir, x))
    ]
    print(sub_dirs)
    if len(sub_dirs) == 0:
        json_paths = glob.glob(os.path.join(json_dir, "*.json"))
    else:
        json_paths = glob.glob(os.path.join(json_dir, "*", "*.json"))
    print(json_paths)
    # for scan_dir in ['test_dir/01']  # 处理单个文件夹
    # f or scan_dir in glob.glob('test_dir/*'): # 处理test_dir下所有文件夹
    for name in json_paths:
        file_path = name
        print(file_path)
        # if os.path.isfile(file_path) and name.split('.')[1] == 'json': # 判断是json文件
        os.system(str("labelme_json_to_dataset " + file_path))
        print("success json to dataset: ", file_path)


if __name__ == '__main__':
    print("注意：如处理单个文件夹当前文件夹下子文件夹应为空")
    print('help: python pipeline.py --json_dir')
    args = init_args()
    main(args.json_dir)
