python finetune_main.py \
    --downstream_dataset 'TUAB' \
    --num_of_classes 6 \
    --model_dir './models_my' \
    --num_workers 4 \
    --datasets_dir "../BigDownstream/TUAB" \
    --cuda 0 \
    --lr 0.001 \
    --multi_lr False \
    --epochs 50 \
    --weight_decay 1e-4 \
    --clip_value 1




#0.86815
#
#
#
#