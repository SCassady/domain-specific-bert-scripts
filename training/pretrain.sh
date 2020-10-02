#!/usr/bin/env bash


# Further pretrains existing BERT models. In this case, Bert-BASE was used.
# It should be okay using BERT's vocab file, even though I used NCBI's here (there may not be a difference).

# create_pretraining_data.py and run_pretraining.py from https://github.com/google-research/bert

# Replace input_file, output_file, output_dir, etc. values

# Takes a single argument:
# Output directory

OUTPUT_DIR="$1"

export BERT_MODEL_DIR="/disk/data2/radiology/users/scassady/bluebert/bert"
export NCBI_MODEL_DIR="/disk/data2/radiology/users/scassady/ncbi"
export BASE_DIR="/disk/data2/radiology/users/scassady/bert-BASE"
export DATA_DIR="/disk/data2/radiology/users/scassady/data"
export PATH="$PATH:/opt/cuda-9.0.176.1/bin"
#export CUDA_PATH=/opt/cuda-9.0.176.1
#export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH


mkdir -p $DATA_DIR/pretraining_output/$OUTPUT_DIR

python $BERT_MODEL_DIR/create_pretraining_data.py \
 	--input_file=$DATA_DIR/processed/mimic-cxr.txt \
 	--output_file=$DATA_DIR/processed/mimic-cxr.tfrecord \
   --vocab_file=$NCBI_MODEL_DIR/vocab.txt \
 	--do_lower_case=True \
 	--max_seq_length=128 \
 	--max_predictions_per_seq=20 \
 	--masked_lm_prob=0.15 \
 	--random_seed=12345 \
 	--dupe_factor=5

python $BERT_MODEL_DIR/run_pretraining.py \
	--input_file=$DATA_DIR/processed/mimic-cxr.tfrecord \
	--output_dir=$DATA_DIR/pretraining_output/$OUTPUT_DIR \
	--do_train=True \
	--do_eval=True \
	--bert_config_file=$BASE_DIR/bert_config.json \
	--init_checkpoint=$BASE_DIR/bert_model.ckpt \
	--train_batch_size=32 \
	--max_seq_length=128 \
	--max_predictions_per_seq=20 \
	--num_train_steps=100000 \
	--num_warmup_steps=10000 \
	--learning_rate=5e-5