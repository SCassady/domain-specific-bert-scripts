#!/bin/sh

#!/usr/bin/env bash


#OUTPUT_DIR="$1"
#CHECKPOINT_DIR="$2"
#export BERT_MODEL_DIR="/disk/data2/radiology/users/scassady/bluebert/bert"


TSV_PATH="/disk/data2/radiology/users/scassady/data/finetuning/ess_grant-merge/"
OUTPUT_PATH="/disk/data2/radiology/users/scassady/data/finetuning_output/merge/ess/bert/"
DATA_PATH="/disk/data2/radiology/users/scassady/data/"
TAY_PATH="/disk/data2/radiology/users/scassady/data/finetuning_output/merge/tay/bert/"
ESS_PATH="/disk/data2/radiology/users/scassady/data/finetuning_output/merge/ess/bert/"
#mkdir -p $OUTPUT_PATH/base
#mkdir -p $DATA_PATH/eval
#mkdir -p $DATA_PATH/eval/ess
#mkdir -p $DATA_PATH/eval/tay
#
#echo "/////////////////////////  (ESS) STARTING BASE//////////////////////////"


#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/ess_grant-merge/test.tsv $ESS_PATH/base $DATA_PATH/eval/ess/base.tsv ess
#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/ess_grant-merge/test.tsv $ESS_PATH/base_cxr $DATA_PATH/eval/ess/base_cxr.tsv ess
#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/ess_grant-merge/test.tsv $ESS_PATH/base_cxr_tay $DATA_PATH/eval/ess/base_cxr_tay.tsv ess
#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/ess_grant-merge/test.tsv $ESS_PATH/base_tay $DATA_PATH/eval/ess/base_tay.tsv ess
#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/ess_grant-merge/test.tsv $ESS_PATH/blue $DATA_PATH/eval/ess/blue.tsv ess
#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/ess_grant-merge/test.tsv $ESS_PATH/domain $DATA_PATH/eval/ess/domain.tsv ess
#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/ess_grant-merge/test.tsv $ESS_PATH/task $DATA_PATH/eval/ess/task.tsv ess
#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/ess_grant-merge/test.tsv $ESS_PATH/blue_tay $DATA_PATH/eval/ess/blue_tay.tsv ess
##
#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/tay-merge/test.tsv $TAY_PATH/base $DATA_PATH/eval/tay/base.tsv tay
#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/tay-merge/test.tsv $TAY_PATH/base_cxr $DATA_PATH/eval/tay/base_cxr.tsv tay
#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/tay-merge/test.tsv $TAY_PATH/base_cxr_tay $DATA_PATH/eval/tay/base_cxr_tay.tsv tay
#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/tay-merge/test.tsv $TAY_PATH/base_tay $DATA_PATH/eval/tay/base_tay.tsv tay
#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/tay-merge/test.tsv $TAY_PATH/blue $DATA_PATH/eval/tay/blue.tsv tay
#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/tay-merge/test.tsv $TAY_PATH/domain $DATA_PATH/eval/tay/domain.tsv tay
#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/tay-merge/test.tsv $TAY_PATH/task $DATA_PATH/eval/tay/task.tsv tay
#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/tay-merge/test.tsv $TAY_PATH/blue_tay $DATA_PATH/eval/tay/blue_tay.tsv tay


#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/ess_grant-merge/test.tsv $ESS_PATH/over_extreme $DATA_PATH/eval/ess/over_extreme_base_cxr_tay.tsv ess
#python experiments/scripts/eval_average.py $DATA_PATH/finetuning/ess_grant-merge/test.tsv $ESS_PATH/oversampled_mod $DATA_PATH/eval/ess/oversampled_mod_base_cxr_tay.tsv ess


python experiments/scripts/eval_average.py $DATA_PATH/finetuning/ess_grant-merge/test.tsv $ESS_PATH/bct_oversampled_10 $DATA_PATH/eval/ess/bct_oversampled_10.tsv ess
python experiments/scripts/eval_average.py $DATA_PATH/finetuning/ess_grant-merge/test.tsv $ESS_PATH/bct_oversampled_25 $DATA_PATH/eval/ess/bct_oversampled_25.tsv ess
python experiments/scripts/eval_average.py $DATA_PATH/finetuning/ess_grant-merge/test.tsv $ESS_PATH/bct_oversampled_50 $DATA_PATH/eval/ess/bct_oversampled_50.tsv ess
python experiments/scripts/eval_average.py $DATA_PATH/finetuning/ess_grant-merge/test.tsv $ESS_PATH/bct_over_extreme $DATA_PATH/eval/ess/bct_oversampled_100.tsv ess

echo "+++++++++++++++++++++++++++++++++++++++++COMPLETE!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"