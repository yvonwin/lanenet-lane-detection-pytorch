source ~/miniconda3/bin/activate pytorch
python train.py --epoch 300 --dataset ./dataset_3   --save nas/enet/log_dataset_3 --model_type ENet --num_worker 16 --bs 16
