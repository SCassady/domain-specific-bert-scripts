#!/bin/sh
#!/usr/bin/env bash

# Example of shell script for finetuning.
# Used for fine-tuning N seeds. (ie; written here is seeds 41 to 50).

# pretraining model locations:
BERT_TASK_PATH="/disk/data2/radiology/users/scassady/data/pretraining_output/long_tay0/pytorch/"

# hyperparameters:
EPOCHS=25
LEARNING_RATE=7e-5

# finetune.py args:
#1.) DATA_PATH: path to directory containing train/eval/test data. Data files would be: train.tsv, dev.tsv, test.tsv.
#2.) BERT_PRETRAINED_PATH: path to directory containing prerained pytorch model, as well as vocab file and bert config:
#  Should contain: bert_config.json  pytorch_model.bin  vocab.txt
#3.) OUTPUT_PATH: path to directory where results and resutling model outputted.
#4.) NUM_EPOCHS: number of finetuning epochs.
#5.) LEARNING_RATE: learning rate, can be specified as "7e-5", etc.
#6.) SEED: random seed, an int.
#7.) EVAL: perform evaluation? 1 = true, 0 = false. Requires dev.csv in data folder.
#8.) PREDICT: predict on test data? 1 = true, 0 = false. Requires test.csv in data folder.

TSV_PATH="/disk/data2/radiology/users/scassady/data/finetuning/over_extreme/"
OUTPUT_PATH="/disk/data2/radiology/users/scassady/data/finetuning_output/merge/ess/bert/"

mkdir -p $OUTPUT_PATH/output_directory

echo "/////////////////////////  (ess) STARTING OVERSAMPLED_MODIFIED-100 //////////////////////////"
for i in {41..50}
do
	echo "=============================================================="
	echo $i
	mkdir -p $OUTPUT_PATH/output_directory/$i
	echo "----------------------------TRAIN------------------------------"
	python finetune.py \
	  $TSV_PATH \
	  $BERT_TASK_PATH \
	  $OUTPUT_PATH/output_directory/$i \
	  $EPOCHS \
	  $LEARNING_RATE \
	  $i \
	  0 \
	  1

done
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@  (ess) FINISHING OVERSAMPLED_MODIFIED-100 @@@@@@@@@@@@@@@@@@@@@@@@@@@"

echo "+++++++++++++++++++++++++++++++++++++++++COMPLETE!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"