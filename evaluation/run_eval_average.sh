#!/bin/sh

#!/usr/bin/env bash

# Example script for running eval_average.py

DATA_PATH="/disk/data2/radiology/users/scassady/data/"
ESS_PATH="/disk/data2/radiology/users/scassady/data/finetuning_output/merge/ess/bert/"

python experiments/scripts/eval_average.py \
  $DATA_PATH/test.tsv \
  $ESS_PATH/bct_oversampled_10 \
  $DATA_PATH/bct_oversampled_10.tsv \
  ess


echo "+++++++++++++++++++++++++++++++++++++++++COMPLETE!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"