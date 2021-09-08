#!/bin/sh
#================================================================
#   Don't go gently into that good night. 
#   
#   author: klaus
#   description: 
#
#================================================================

dataset=cifar100
NUM_OF_GPU=8
BATCH_SIZE_PER_GPU=32
python3 -m torch.distributed.launch --nproc_per_node=$NUM_OF_GPU train.py \
    --train_batch_size $BATCH_SIZE_PER_GPU --name $dataset-100_500 --dataset $dataset \
    --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  \
    --gradient_accumulation_steps 2
