#source ~/miniconda3/bin/activate pytorch16
python test_lanenet.py --model_type ENet --src_dir './data/test_img/03/' --model 'log/log_example/2021-12-03-14-16-21_epochs10_ENet__best_model.pth'