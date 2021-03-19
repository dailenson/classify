# Cifar10 with WRN 🌁

This folder contains a simple Wide-ResNet implementation that can be trained on Cifar10 with SAM. Start the training by running `python3 train.py`

- bash ./train_cfair_0001.sh 使用lr=0.001的配置在cfair训练SAM方法
- bash ./train_cfair_autoaugment.sh 使用autoaugment数据增强方式在cfair训练SAM方法
- bash ./train_cfair_wrn_28_10.sh 使用wrn_28_10作为backbone在cfair训练SAM方法
- bash ./train_cfair_wrn_52_12.sh 使用wrn_52_12作为backbone在cfair训练SAM方法
- bash ./train_cfair.sh 使用默认配置在cfair上练SAM方法
- bash ./train_minist.sh 使用默认配置在minist上练SAM方法