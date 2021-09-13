#!/bin/sh
#================================================================
#   Don't go gently into that good night. 
#   
#   author: klaus
#   description: 
#
#================================================================

#dataset=cifar100
#name=$dataset-100_500-run1
#python test.py  --name $name --dataset $dataset  \
#    --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  \
#    --gradient_accumulation_steps 2

### with rw_v2
#dataset=cifar100
#name=$dataset-100_500-run1
#python test.py  --name $name --dataset $dataset  \
#    --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  \
#    --alpha 0.1 --rw --rw_version v2

### with rw_v2
#dataset=cifar100
#name=$dataset-100_500-run1
#python test.py  --name $name --dataset $dataset  \
#    --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  \
#    --alpha 0.1 --rw --rw_version v2

## with rw_v3
dataset=cifar100
name=$dataset-100_500-run1
python test.py  --name $name --dataset $dataset  \
    --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  \
    --alpha 0.5 --rw --rw_version v3
