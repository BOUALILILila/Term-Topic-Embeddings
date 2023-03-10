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
# context window (ws*2+1)
w=3 # one token to the right/left: t[-1] t[0] t[+1]
# Batch size
BATCH=2048
# Learning Rate
LR=2e-3
# Wikipedia passages # {pid}\t{text}
# 20561327 passages in total / 2048 ~= 10040 ~= 1 epoch
# 100400 steps ~= 10 epochs
DATADIR="."
DATAFN=car-windows_128_96_bert-base-uncased.tsv
# Sample uniform
strategy="uni"

OUTDIR="."
RUN_NAME="frozen-gpu_${PRETRAINED##*/}_b${BATCH}_N${N}_K${K}_W${w}_lr${LR}_SPL${strategy}_TTM_LCM_${DATAFN%.tsv}"
###

python  train/pre-train/pre_train_SenseEmbedWithAttnSGU--TTM-LCM.py \
    --train_path ${DATADIR}/$DATAFN \
    --max_seq_len $MAXSEQLEN \
    --N $N \
    --sampling_strategy $strategy \
    --model_name_or_path $PRETRAINED \
    --num_senses $K \
    --use_value true \
    --is_cross true \
    --sliding_context_window $w \
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
    --max_steps 100400 \
    --save_strategy steps \
    --save_steps 20080 \
    --freeze_transformer_weights \
    --fp16 \
    --report_to ${integrations} 

