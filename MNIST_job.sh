#!/bin/bash

# PIPNet directory
cd /pfs/data5/home/ma/ma_ma/ma_ajoseph/PIPNet

# Activate environment
source /pfs/data5/home/ma/ma_ma/ma_ajoseph/PIPMIL/thesis_env/bin/activate

# # Run with pretrained model
# python main.py --dataset 'CUB-200-2011' --epochs_pretrain 0 --batch_size 64 --freeze_epochs 10 --epochs 0 --log_dir './runs/pipnet_cub' --state_dict_dir_net './pipnet_cub_trained'

# Run your Python script with the specified arguments
python main.py --dataset 'MNIST' --validation_size 0.0 --net 'convnext_tiny_26' --batch_size 64  --batch_size_pretrain 128 --epochs 10 --optimizer 'Adam' --lr 0.05 --lr_block 0.0005 --lr_net 0.0005 --weight_decay 0.0 --log_dir './runs/pipnet_mnist_cnext26' --num_features 0 --image_size 112 --state_dict_dir_net '' --freeze_epochs 10 --dir_for_saving_images 'Visualization_results' --epochs_pretrain 10 --seed 1 --gpu_ids '' --num_workers 8 
