CUDA_VISIBLE_DEVICES=5 python train.py \
--learning_rate=0.001
--dataset="cfair"
--channel=3
--resize_size=32
--original_size=32
--autoaugment=0