#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 添加命令行参数
SEED=43
EPOCHS=200
LR=0.01
WEIGHT_DECAY=5e-4
HIDDEN=16
DROPOUT=0.3

# 运行Python脚本
python train.py --seed $SEED \
                --epochs $EPOCHS \
                --lr $LR \
                --weight_decay $WEIGHT_DECAY \
                --hidden $HIDDEN \
                --dropout $DROPOUT