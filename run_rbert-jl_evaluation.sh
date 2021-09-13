#!/usr/bin/env bash
# 
#PBS -l select=1:ncpus=1:ngpus=1:mem=5G
#PBS -l walltime=00:30:59

cd "${PBS_O_WORKDIR}"
cd ../../../../medical_transformer

# PARAMS
export TASK_NAME=cdcp_relclass_rbert_jl
export MODELTYPE=bert-rbert-jl-2
export CLASSIFIER=None
export MAXSEQLENGTH=512
export BATCH_SIZE=8
export LEARNING_RATE=2e-5
export WEIGHT_DECAY=0
export FREEZE_BERT=0
export EPOCHS=20
export OUTPUTDIR=output/$MODELTYPE+$CLASSIFIER+$TASK_NAME+$MAXSEQLENGTH+LR$LEARNING_RATE+WD$WEIGHT_DECAY+FZ$FREEZE_BERT+EP$EPOCHS+BS$BATCH_SIZE

# PATH TO TRAINING DATA
export DATA_DIR=data/cdcp/original

# SELECT MODEL FOR FINE-TUNING

export MODEL=$OUTPUTDIR
#export MODEL=bert-base-uncased
#export MODEL=monologg/biobert_v1.1_pubmed
#export MODEL=monologg/scibert_scivocab_uncased
#export MODEL=allenai/scibert_scivocab_uncased
#export MODEL=roberta-base

#PBS -N $TASK_NAME+CDCP+$MAXSEQLENGTH+bert
#PBS -o $TASK_NAME+CDCP+$MAXSEQLENGTH+bert_out

#python train_multiplechoice.py \
python3 train.py \
  --model_type $MODELTYPE \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUTDIR \
  --task_name $TASK_NAME \
  --classifier_type $CLASSIFIER \
  --loss_weights 0.00115 0.03788 1 \
  --do_eval \
  --freeze_bert $FREEZE_BERT \
  --evaluate_during_training \
  --do_lower_case \
  --data_dir $DATA_DIR \
  --max_seq_length $MAXSEQLENGTH \
  --overwrite_output_dir \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $EPOCHS \
  --save_steps 1000 \
  --weight_decay $WEIGHT_DECAY \
