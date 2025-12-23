#!/bin/bash

# Fine-tuning script for TUEV dataset
# Description: This script fine-tunes a model on the TUEV dataset with 6 classes

python finetune_main.py \
    --downstream_dataset 'CHB-MIT' \
    --num_of_classes 1 \
    --model_dir './models_my' \
    --num_workers 4 \
    --datasets_dir "../BigDownstream/chb-mit/processed_seg" \
    --cuda 0 \
    --lr 0.0005 \
    --multi_lr False \
    --epochs 5 \
    --weight_decay 1e-4 \
    --clip_value -1