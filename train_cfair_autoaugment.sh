CUDA_VISIBLE_DEVICES=0 python train.py \
--dataset="cifar"
--channel=3
--resize_size=32
--original_size=32
--autoaugment=1