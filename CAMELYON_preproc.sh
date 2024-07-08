#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=100gb 
#SBATCH --job-name=slurm-preproc_camelyon_resnet
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-preproc_camelyon_resnet.out

# PIPNet directory
cd /pfs/work7/workspace/scratch/ma_ajoseph-ProtoData/ma_ajoseph/PIPNet

# Activate environment
#source /pfs/work7/workspace/scratch/ma_ajoseph-ProtoData/ma_ajoseph/PIPMIL/thesis_env/bin/activate
conda activate thesis_env

# # Run with pretrained model
# python main.py --dataset 'CUB-200-2011' --epochs_pretrain 0 --batch_size 64 --freeze_epochs 10 --epochs 0 --log_dir './runs/pipnet_cub' --state_dict_dir_net './pipnet_cub_trained'

# Run your Python script with the specified arguments
# python main.py --dataset 'CUB-200-2011' --validation_size 0.0 --net 'convnext_tiny_26' --batch_size 64  --batch_size_pretrain 128 --epochs 60 --optimizer 'Adam' --lr 0.05 --lr_block 0.0005 --lr_net 0.0005 --weight_decay 0.0 --log_dir './runs/pipnet_cub_cnext26_3' --num_features 0 --image_size 224 --state_dict_dir_net '' --freeze_epochs 10 --dir_for_saving_images 'Visualization_results' --epochs_pretrain 10 --seed 1 --gpu_ids '' --num_workers 8 

# PRE-TRAINING
# python main.py --dataset 'CAMELYON' --validation_size 0.0 --net 'resnet18' --batch_size 5  --batch_size_pretrain 128 --epochs 60 --optimizer 'Adam' --lr 0.05 --lr_block 0.0005 --lr_net 0.0005 --weight_decay 0.0 --log_dir './runs/PIPMIL_CAMELYON_pretrain_resnet18' --num_features 0 --image_size 224 --state_dict_dir_net '' --freeze_epochs 10 --dir_for_saving_images 'Visualization_results' --epochs_pretrain 10 --seed 631056511 --gpu_ids '' --num_workers 8 --bias

# TRAINING
# python main.py --dataset 'CAMELYON' --validation_size 0.0 --net 'resnet18' --batch_size 5  --batch_size_pretrain 128 --epochs 60 --optimizer 'Adam' --lr 0.05 --lr_block 0.0005 --lr_net 0.0005 --weight_decay 0.0 --log_dir './runs/PIPMIL_CAMELYON_train_resnet_3' --num_features 0 --image_size 224 --state_dict_dir_net '' --freeze_epochs 10 --dir_for_saving_images 'Visualization_results' --epochs_pretrain 0 --seed 631056511 --gpu_ids '' --num_workers 8 --bias 

# PREPROCESSING SAMPLES
python3 util/camelyon_resnet.py

# mv './runs/PIPMIL_CAMELYON_train_whole_2/log_epoch_overview.csv' "./runs/PIPMIL_CAMELYON_train_whole_2/log_epoch_overview1.csv"

# for i in {2..6}
# do
#     echo "Running iteration $i"
#     python main.py --dataset 'CAMELYON' --validation_size 0.0 --net 'convnext_tiny_26' --batch_size 2  --batch_size_pretrain 128 --epochs 10 --optimizer 'Adam' --lr 0.05 --lr_block 0.0005 --lr_net 0.0005 --weight_decay 0.0 --log_dir './runs/PIPMIL_CAMELYON_train_whole_2' --num_features 0 --image_size 224 --state_dict_dir_net './runs/PIPMIL_CAMELYON_train_whole_2/checkpoints/net_trained' --freeze_epochs 10 --dir_for_saving_images 'Visualization_results' --epochs_pretrain 0 --seed 631056511 --gpu_ids '' --num_workers 8 --bias
#     mv './runs/PIPMIL_CAMELYON_train_whole_2/log_epoch_overview.csv' "./runs/PIPMIL_CAMELYON_train_whole_2/log_epoch_overview$i.csv"
#     # python main.py --dataset 'nabirds' --validation_size 0.0 --net 'convnext_tiny_26' --batch_size 64  --batch_size_pretrain 128 --epochs 60 --optimizer 'Adam' --lr 0.05 --lr_block 0.0005 --lr_net 0.0005 --weight_decay 0.0 --log_dir './runs/pipnet_cub_cnext26_1' --num_features 0 --image_size 224 --state_dict_dir_net './runs/pipnet_cub_cnext26_1/checkpoints/net_trained' --freeze_epochs 10 --dir_for_saving_images 'Visualization_results' --epochs_pretrain 10 --seed 1 --gpu_ids '' --num_workers 8 
# done