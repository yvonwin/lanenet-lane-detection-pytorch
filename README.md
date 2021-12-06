# Lanenet-Lane-Detection (基于pytorch的版本)

## 简介     
在本项目中，使用pyotrch复现了 IEEE IV conference 的论文 "Towards End-to-End Lane Detection: an Instance Segmentation Approach"，并对这篇论文的思想进行讨论。   
开发这一项目的初衷是，在github上开源的LaneNet项目数目较少，其中只有基于tensorflow 1.x的项目https://github.com/MaybeShewill-CV/lanenet-lane-detection 能够完整的实现作者论文中的思想，但是随着tensorflow 2.x的出现，基于tensorflow 1.x的项目在未来的维护将会越来越困难，很多刚入门深度学习同学也不熟悉tensorflow 1.x的相关功能。与此同时，github上基于pytorch的几个LaneNet项目或多或少都存在一些错误，例如错误复现Discriminative loss导致实例分割失败，且相关作者已经不再维护。   

LaneNet的网络框架:    
![NetWork_Architecture](./data/source_image/network_architecture.png)

## 下载项目

```
git clone https://github.com/yvonwin/lanenet-lane-detection-pytorch.git
```

## 安装依赖

到torch官网安装torch 1.6到1.9都可以和torchvision

安装其他依赖

```
pip installl -r requirements
```



## 生成用于训练和测试的Tusimple车道线数据集

在此处下载Tusimple数据集： [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3).  
运行以下代码生成可以用于训练的数据集形式： 
仅生成训练集：   
```
python tusimple_transform.py --src_dir path/to/your/unzipped/file --val False
```
生成训练集+验证集:    
```
python tusimple_transform.py --src_dir path/to/your/unzipped/file --val True
```
生成训练集+验证集+测试集:    
```
python tusimple_transform.py --src_dir path/to/your/unzipped/file --val True --test True
```
path/to/your/unzipped/file应该包含以下文件:    
```
|--dataset
|----clips
|----label_data_0313.json
|----label_data_0531.json
|----label_data_0601.json
|----test_label.json
```

## 自定义数据转换成tusimle格式数据

1. 视频转换图片，使用quick cut或ffmpeg或Python脚本皆可

   python脚本为例

   local_utils/videos2pictures.py

2. labelme人工标注
   标注数据格式linesrip

   主要结构分析

   ```
   ├── v1_1.jpg
   ├── v1_1.json
   ```

   使用labelme_json_to_dataset转换单张

   ```
   labelme_json_to_dataset v1_1.json
   ```

   会生成一个文件夹名为v1_1_json

   ```
   (base) ➜  v1_1_json tree
   .
   ├── img.png
   ├── label.png
   ├── label_names.txt
   └── label_viz.png
   ```

   

3. 批量转换json_to_dataset

   local_utils/pipeline.py

4. 转换成tusimple

   local_utils/my_transform.py

   

##	训练模型

在示例的文件夹中运行训练代码，注意，使用示例数据并不能真正训练出可以用的LaneNet，因为示例文件只有6张图像:   
```
python train.py --epoch 10 --dataset ./data/training_data_example  --save ./log/log_example --model_type ENet --num_worker 16 --bs 16
```


## 测试

使用我在NVIDIA RTX 2070上训练好的模型，仅仅训练了25个epoch，但已经具备一定的预测效果。         
测试模型，以示例数据的测试图像为例:    
```
python test.py --img ./data/tusimple_test_image/0.jpg
```

批量测试文件夹图片

```
python test_lanenet.py --model_type ENet --src_dir './data/test_img/03/' --model 'log/log_example/2021-12-03-14-16-21_epochs10_ENet__best_model.pth'
```

测试rtsp流

```
python test_lanenet_rtsp.py --model ../lanenet-pytorch_models/2021-08-18-14-10-53_epochs300_ENet_resnet101_best_model.pth  --model_type ENet --rtsp_url rtsp://admin:Zhaodao1234!@192.168.31.63:554//Streaming/Channels/2
```
如果手上没有相机，需要模拟rtsp流，可以使用rtsp-simple-server和ffmpeg来完成。
```
./rtsp-simple-server
```
车道线视频为test.mp4
```
ffmpeg -re -r 25 -i test.mp4 -codec copy -an -f rtsp -muxdelay 0  -rtsp_transport tcp rtsp://localhost:8554/
```
修改rtsp流为自定义的rtsp://localhost:8554/即可



## 讨论分析:    
更新日志：    
2021.7.16更新    
增加了DeepLabv3+作为LaneNet的Encoder和Decoder,实际效果未测试。    

2021.7.22更新    
增加了Focal Loss。     

我的工作。

待更新

2021.8.23 项目主体结构完成

三次标注数据汇总训练完成。

目前项目中ENet表现最好

此项目待完善部分
- post process ✅
- 参数优化 参数加载checkpoint 参数加载模型 ✅
- 车道线目标限界问题
