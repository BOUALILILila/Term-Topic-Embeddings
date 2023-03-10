#!/bin/bash

# Activate a conda environment
conda activate Myenv

# tensorboard and/or wandb
integrations='none'

PRETRAINED='colbert-distilbert-margin_mse-T2-msmarco-bertonly'
TOK='distilbert-base-uncased'

# Num tokens per long and short context respectively
N=50
# Seq len 
MAXSEQLEN=128
# num_senses
K=10
# Batch size
BATCH=4096
# Learning Rate
LR=2e-3
# Wikipedia passages # {pid}\t{text}
# 20561327 passages in total / 4096 ~= 5020 ~= 1 epoch
# 50200 steps ~= 10 epochs
DATADIR="."
DATAFN=car-windows_128_96_bert-base-uncased.tsv
# Sample uniform
strategy="uni"

OUTDIR="."
RUN_NAME="frozen-gpu_${PRETRAINED##*/}_b${BATCH}_N${N}_K${K}_lr${LR}_SPL${strategy}_TTM_${DATAFN%.tsv}"
###

python  train/pre-train/pre_train_SenseEmbed--TTM.py \
    --train_path ${DATADIR}/$DATAFN \
    --max_seq_len $MAXSEQLEN \
    --N $N \
    --sampling_strategy $strategy \
    --model_name_or_path $PRETRAINED \
    --num_senses $K \
    --tokenizer_name $TOK \
    --run_name $RUN_NAME \
    --output_dir ${OUTDIR}/${RUN_NAME} \
    --do_train \
    --overwrite_output_dir \
    --dataloader_num_workers 8 \
    --per_device_train_batch_size $BATCH \
    --learning_rate $LR \
    --lr_scheduler_type constant \
    --warmup_ratio 0 \
    --log_level info \
    --logging_strategy steps \
    --logging_steps 200 \
    --max_steps 50200 \
    --save_strategy steps \
    --save_steps 10040 \
    --freeze_transformer_weights \
    --fp16 \
    --report_to ${integrations} 

