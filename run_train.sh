#!/usr/bin/env bash

# ACTIVATE VIRTUAL ENVIRONMENT
env\Scripts\activate.bat

# SELECT TASK

# 1) CDCP SEQUENCE TAGGING
#export TASK_NAME=cdcp_seqtag
#export MODELTYPE=bert-seqtag

# 1) SEQUENCE TAGGING
#export TASK_NAME=seqtag
#export MODELTYPE=bert-seqtag

# 2) RELATION CLASSIFICATION
export TASK_NAME=relclass
export MODELTYPE=bert

# 3) MULTIPLE CHOICE (requires that train_multiplechoice.py is executed instead of train.py, see below)
#export TASK_NAME=multichoice
#export MODELTYPE=bert-multichoice


# PATH TO TRAINING DATA
export DATA_DIR=data/cdcp/sequence_tags

# MAXIMUM SEQUENCE LENGTH
#export MAXSEQLENGTH=128
export MAXSEQLENGTH=128
export OUTPUTDIR=output/$TASK_NAME+CDCP+$MAXSEQLENGTH/


# SELECT MODEL FOR FINE-TUNING

export MODEL=bert-base-uncased
#export MODEL=monologg/biobert_v1.1_pubmed
#export MODEL=monologg/scibert_scivocab_uncased
#export MODEL=allenai/scibert_scivocab_uncased
#export MODEL=roberta-base


#python train_multiplechoice.py \
python3 train.py \
  --model_type $MODELTYPE \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUTDIR \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR \
  --max_seq_length $MAXSEQLENGTH \
  --overwrite_output_dir \
  --per_gpu_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --save_steps 1000 \
  --overwrite_cache #req for multiple choice