#!/usr/bin bash
for ds in 0 10 20
do
  python main.py \
      --dataset banking \
      --divide_seed $ds \
      --seed 0 \
      --n_ind_class 32 \
      --pretrain \
      --proto_m 0.7 \
      --pretrain_batch_size 256 \
      --pretrain_epoch 20 \
      --select_ood_exemplars_method random \
      --freeze_bert_parameters \
      --gpu_id 0 \
      --max_seq_length 55 \
      --wandb_project_name your_project_name \
      --wandb_mode offline

done

for ds in 0 10 20
do
  python main.py \
      --dataset clinc \
      --divide_seed $ds \
      --seed 0 \
      --n_ind_class 32 \
      --pretrain \
      --proto_m 0.7 \
      --pretrain_batch_size 256 \
      --pretrain_epoch 20 \
      --select_ood_exemplars_method random \
      --freeze_bert_parameters \
      --gpu_id 0 \
      --max_seq_length 30 \
      --wandb_project_name DPL_banking_32_dseed_0 \
      --wandb_mode offline

done