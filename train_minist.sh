CUDA_VISIBLE_DEVICES=2 python train.py \
--dataset="minist"
--channel=1
--resize_size=32
--original_size=28
--autoaugment=0
--epochs=50