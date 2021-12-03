source ~/miniforge3/bin/activate pytorch
#source ~/miniconda/bin/activate pytorch
python train.py --epoch 10 --dataset ./data/training_data_example  --save ./log/log_example --model_type ENet --num_worker 16 --bs 16
