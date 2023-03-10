#!/bin/bash

# Activate a conda environment
conda activate Myenv

# tensorboard and/or wandb
integrations='none'
# the pre-trqined TTM-LCM checkpoints path
PRETRAINED='frozen-gpu_colbert-distilbert-margin_mse-T2-msmarco-bertonly_b2048_N50_K10_W3_lr2e-3_SPLuni_TTM_LCM_car-windows_128_96_bert-base-uncased'
ckpt=100400
TOK='distilbert-base-uncased'

# Batch size
BATCH=64
# Learning Rate
LR=2e-5

sim_fct='dot'

DATADIR="data"

OUTDIR="."
RUN_NAME="unfrozen-gpu_${PRETRAINED##*/}_ckpt${ckpt}_b${BATCH}_lr${LR}_linear_warmup${warmup}_sim${sim_fct}_MMSELoss_msmarco"
###

python  train/fine-tune/fine_tune_SenseEmbed--TTM-LCM.py \
    --freeze_subembeddings true \
    --data_folder ${DATADIR} \
    --max_seq_len 256 \
    --max_query_len 30 \
    --model_name_or_path $PRETRAINED/checkpoint-$ckpt \
    --similarity_fct ${sim_fct} \
    --tokenizer_name $TOK \
    --run_name $RUN_NAME \
    --output_dir ${OUTDIR}/${RUN_NAME} \
    --do_train \
    --overwrite_output_dir \
    --dataloader_num_workers 8 \
    --per_device_train_batch_size $BATCH \
    --learning_rate $LR \
    --lr_scheduler_type linear \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --log_level info \
    --logging_strategy steps \
    --logging_steps 200 \
    --max_steps 235755 \
    --save_strategy steps \
    --save_steps 235755 \
    --freeze_transformer_weights \
    --fp16 \
    --report_to ${integrations} 

