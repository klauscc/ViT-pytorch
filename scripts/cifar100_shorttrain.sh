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
BATCH_SIZE_PER_GPU=64

## Baseline
#name=shorttrain-$dataset-baseline
#python3 -m torch.distributed.launch --nproc_per_node=$NUM_OF_GPU train.py \
#    --train_batch_size $BATCH_SIZE_PER_GPU --name $name --dataset $dataset \
#    --num_steps 1000 --warmup_steps 100 \
#    --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  \
#    --gradient_accumulation_steps 2 \

### rw_v1 alpha:0.1
#name=shorttrain-$dataset-rw_1-alpha0.1
#python3 -m torch.distributed.launch --nproc_per_node=$NUM_OF_GPU train.py \
#    --train_batch_size $BATCH_SIZE_PER_GPU --name $name --dataset $dataset \
#    --num_steps 1000 --warmup_steps 100 \
#    --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  \
#    --gradient_accumulation_steps 2 \
#    --rw --alpha 0.1

### rw_v2 alpha:0.1. dub
#version=v2
#NUM_OF_GPU=4
#BATCH_SIZE_PER_GPU=128
#ACCUM_STEPS=4
#name=shorttrain-$dataset-rw_$version-alpha0.1
#CUDA_VISIBLE_DEVICES=0,2,4,6  python3 -m torch.distributed.launch --nproc_per_node=$NUM_OF_GPU train.py \
#    --train_batch_size $BATCH_SIZE_PER_GPU --name $name --dataset $dataset \
#    --num_steps 1000 --warmup_steps 100 \
#    --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  \
#    --gradient_accumulation_steps $ACCUM_STEPS \
#    --rw --alpha 0.1 --rw_version $version

## rw_v2 alpha:0.5. dub
version=v2
NUM_OF_GPU=8
BATCH_SIZE_PER_GPU=64
ACCUM_STEPS=2
name=shorttrain-$dataset-rw_$version-alpha0.5
python3 -m torch.distributed.launch --nproc_per_node=$NUM_OF_GPU train.py \
    --train_batch_size $BATCH_SIZE_PER_GPU --name $name --dataset $dataset \
    --num_steps 1000 --warmup_steps 100 \
    --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  \
    --gradient_accumulation_steps $ACCUM_STEPS \
    --rw --alpha 0.5 --rw_version $version

### rw_v3 alpha:0.1. den
#version=v3
#NUM_OF_GPU=4
#BATCH_SIZE_PER_GPU=128
#ACCUM_STEPS=4
#name=shorttrain-$dataset-rw_$version-alpha0.1
#CUDA_VISIBLE_DEVICES=0,2,4,6  python3 -m torch.distributed.launch --nproc_per_node=$NUM_OF_GPU train.py \
#    --train_batch_size $BATCH_SIZE_PER_GPU --name $name --dataset $dataset \
#    --num_steps 1000 --warmup_steps 100 \
#    --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  \
#    --gradient_accumulation_steps $ACCUM_STEPS \
#    --rw --alpha 0.1 --rw_version $version


### rw_v3_x4 alpha:0.1. eur
#version=v3_x4
#NUM_OF_GPU=4
#BATCH_SIZE_PER_GPU=128
#ACCUM_STEPS=4
#name=shorttrain-$dataset-rw_$version-alpha0.1
#CUDA_VISIBLE_DEVICES=1,3,5,7  python3 -m torch.distributed.launch --nproc_per_node=$NUM_OF_GPU train.py \
#    --train_batch_size $BATCH_SIZE_PER_GPU --name $name --dataset $dataset \
#    --num_steps 1000 --warmup_steps 100 \
#    --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  \
#    --gradient_accumulation_steps $ACCUM_STEPS \
#    --rw --alpha 0.1 --rw_version $version

### rw_v2_x4 alpha:0.1. dan
#version=v2_x4
#NUM_OF_GPU=4
#BATCH_SIZE_PER_GPU=128
#ACCUM_STEPS=4
#name=shorttrain-$dataset-rw_$version-alpha0.1
#CUDA_VISIBLE_DEVICES=0,2,4,6  python3 -m torch.distributed.launch --nproc_per_node=$NUM_OF_GPU train.py \
#    --train_batch_size $BATCH_SIZE_PER_GPU --name $name --dataset $dataset \
#    --num_steps 1000 --warmup_steps 100 \
#    --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz  \
#    --gradient_accumulation_steps $ACCUM_STEPS \
#    --rw --alpha 0.1 --rw_version $version
