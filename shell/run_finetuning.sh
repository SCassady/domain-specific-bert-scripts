#!/bin/sh
#!/usr/bin/env bash

# Used for fine-tuning N seeds. (ie; written here is seeds 41 to 50).

# pretraining model locations:
BERT_TASK_PATH="/disk/data2/radiology/users/scassady/data/pretraining_output/long_tay0/pytorch/"

# hyperparameters:
EPOCHS=25
LEARNING_RATE=7e-5

# finetune.py args:
# 1) finetuning input data (folder with train, dev, test .tsv)
# 2) pretrained pytorch model
# 3) output folder
# 4) number of epochs
# 5) learning rate
# 6) random seed
# 7) do evaluation?
# 8) do prediction?

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