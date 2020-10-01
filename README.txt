Dependencies:

- BERT needs to be installed, from: [url]
- For use with NCBI-BERT, it must be installed from: [url]


Notes:

- paths were mostly hard-coded for my directories, so most will have to be changed.








For Steven Cassady's 2020 MSc Submission; August 21, 2020;
This folder contains the main scripts (ignoring small shell scripts):


count_most_frequent:
used to count and output the most common words in data sets.

count_tokens:
used to count mean, median, max # of tokens per report, to ensure fine-tuning parameters are acceptable.

count_labels:
used to verify # of reports of each label for fine-tuning data sets.

convert_to_torch:
converts pre-trained tensorflow model to pytorch. original script 

eval_average:
calculates metrics for 10 random initializations, averages them and outputs to tsv.

oversample:
oversamples a dataset using ML-ROS.

partition_ess_grant:
used to partition training set into 5 folds.

torch_classifier:
performs BERT fine-tuning, using pytorch. largely inspired by https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d .

train_cxr:
performs BERT pre-training using tensorflow. can be modified for any data set.

tsvify_ess_grant, tsvify_tay:
used to shape data into format suitable for fine-tuning BERT models. can be modified for any data set.

process_cxr, process_tay:
used to prepare data for pre-training. modified from script shared by Andreas Grivas.

gridsearch_ess:
used for searching over learning rates, epochs. can be modified for any data set.

puretrain_ess:
used for final training.  can be modified for any data set.