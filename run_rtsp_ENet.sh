#source ~/miniconda3/bin/activate pytorch16
echo "version-1.0"
# python test_lanenet_rtsp.py --model ../lanenet-pytorch_models/2021-08-18-14-10-53_epochs300_ENet_resnet101_best_model.pth  --model_type ENet --rtsp_url rtsp://admin:Zhaodao1234!@192.168.31.63:554//Streaming/Channels/2
python test_lanenet_rtsp.py --model ../lanenet-pytorch_models/2021-08-18-14-10-53_epochs300_ENet_resnet101_best_model.pth  --model_type ENet --rtsp_url rtsp://127.0.0.1:8554/mystream