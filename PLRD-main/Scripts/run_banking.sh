#!/usr/bin bash

# OOD Ratio=40%
for ds in 0 10 20
do
  python main.py \
      --dataset banking \
      --divide_seed $ds \
      --seed 4 \
      --n_ind_class 47 \
      --pretrain \
      --train \
      --n_exemplar_per_class 5 \
      --train_batch_size 256\
      --train_epoch 200 \
      --proto_m 0.7 \
      --lr_train 0.01 \
      --freeze_bert_parameters \
      --gpu_id 0 \
      --use_memory \
      --max_seq_length 55 \
      --wandb_project_name your_project_name \
      --wandb_mode offline
done


# OOD Ratio=60%
for ds in 0 10 20
do
  python main.py \
      --dataset banking \
      --divide_seed $ds \
      --seed 4 \
      --n_ind_class 32 \
      --pretrain \
      --train \
      --n_exemplar_per_class 5 \
      --train_batch_size 256\
      --train_epoch 200 \
      --proto_m 0.7 \
      --lr_train 0.01 \
      --freeze_bert_parameters \
      --gpu_id 0 \
      --use_memory \
      --max_seq_length 55 \
      --wandb_project_name your_project_name \
      --wandb_mode offline
done

# OOD Ratio=80%
for ds in 0 10 20
do
  python main.py \
      --dataset banking \
      --divide_seed $ds \
      --seed 4 \
      --n_ind_class 17 \
      --pretrain \
      --train \
      --n_exemplar_per_class 5 \
      --train_batch_size 256\
      --train_epoch 200 \
      --proto_m 0.9 \
      --lr_train 0.01 \
      --freeze_bert_parameters \
      --gpu_id 0 \
      --use_memory \
      --max_seq_length 55 \
      --wandb_project_name your_project_name \
      --wandb_mode offline
done




