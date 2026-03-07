#!/usr/bin/env bash

docker run -it -v "/features-2/MVNM_Training_Data":/train \
-v "/features-2/MVNMv2.5_TrainingImages":/test \
-v "$(pwd)/save":/save \
--runtime=nvidia --user 1000:1000 \
features python train_test.py \
-lr --train=/train --test=/test --save=/save \
--name=vgg16_1024 --layer=block2_conv2
