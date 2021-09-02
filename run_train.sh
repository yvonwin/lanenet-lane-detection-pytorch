source ~/miniconda3/bin/activate pytorch
###
 # @Author: your name
 # @Date: 2021-08-30 10:14:23
 # @LastEditTime: 2021-08-30 10:16:50
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /lanenet-lane-detection-pytorch/run_train.sh
### 
# python test_lanenet_rtsp.py --model ~/Desktop/models/2021-08-18-14-10-53_epochs300_ENet_resnet101_best_model.pth  --model_type ENet --rtsp_url rtsp://admin:Zhaodao1234!@192.168.31.63:554//Streaming/Channels/2
 python train.py --epoch 300 --dataset ./dataset_3   --save nas/enet/log_dataset_3 --model_type ENet --num_worker 16 --bs 16
