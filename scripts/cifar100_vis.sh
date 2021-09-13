#!/bin/sh
#================================================================
#   Don't go gently into that good night. 
#   
#   author: klaus
#   description: 
#
#================================================================

## with rw_v3
dataset=cifar100
name=$dataset-100_500-run1
python visualize.py  --name $name --dataset $dataset  \
    --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  \
    --vis
