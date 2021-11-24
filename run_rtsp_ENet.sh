#source ~/miniconda3/bin/activate pytorch16
###
 # @Author: your name
 # @Date: 2021-08-23 17:50:32
 # @LastEditTime: 2021-11-24 14:33:47
 # @LastEditors: Please set LastEditors
 # @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 # @FilePath: /lanenet-lane-detection-pytorch/run_rtsp_ENet.sh
### 
#source ~/miniconda3/bin/activate pytorch16
# python test_lanenet_rtsp.py --model ../lanenet-pytorch_models/2021-08-18-14-10-53_epochs300_ENet_resnet101_best_model.pth  --model_type ENet --rtsp_url rtsp://admin:Zhaodao1234!@192.168.31.63:554//Streaming/Channels/2
python test_lanenet_rtsp.py --model ../lanenet-pytorch_models/2021-08-18-14-10-53_epochs300_ENet_resnet101_best_model.pth  --model_type ENet --rtsp_url rtsp://127.0.0.1:8554/mystream