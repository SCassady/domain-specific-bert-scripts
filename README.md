## BERT domain-specific training scripts for use with Multi-label classification

Scripts to further pretrain pretrained bert models on more data, and then to finetune on a multi-label classification task. Also includes evaluation scripts and counting scripts, but some are currently hardcoded for the ESS and Tayide datasets.

In order to use this, the following should also be installed:
- the official bert repository: https://github.com/google-research/bert
- a pretrained bert-base model; for example, "BlueBERT-Base, Uncased, PubMed+MIMIC-III" from https://github.com/ncbi-nlp/bluebert

### Environments

For the MSc work, 2 environments were used.

The pretraining env included:
- python                    3.6.10  
- bert-tensorflow           1.0.1  
- tensorflow                1.11.0        
- tensorflow-base           1.11.0        
- tensorflow-gpu            1.11.0     
- lxml                      4.5.1 (for creating tsv files)
- nltk                      3.5 (if counting tokens)

The finetuning environment, also used for converting the tensorflow model to pytorch included:
- python                    3.6.10 
- tensorflow              2.2.0
- transformers            3.0.1
- pytorch-pretrained-bert 0.6.2
- torch                   1.5.1
- numpy                   1.19.0
- pandas                  1.0.5
- tqdm                      4.47.0
- scikit-learn              0.23.1
- psutil                    5.7.2 (only used for benchmarking in oversampling scripts)
- nltk                      3.5 (if counting tokens)

The full list of packages can be seen in finetuning_package_list.txt and pretraining_conda_package_list.txt

## Pre-processing

The following steps were used for preparing the ESS, MIMIC-CXR, and Tayside data:

1. Prepare the reports for bert pretraining with bertify_cxr.py and bertify_tay.py (producing .txt files with sentences each on their own lines.)
2. Prepare the reports for multi-label fine-tuning with tsvify_tay.py, tsvify_ess_grant.py. The resulting files were .tsv files, with the first columns being ID and the text content, followed by columns for each of the labels (taking 1 or 0 values).
3. The training set was partitioned for 5-fold cross-validation. This isn't necessary for using the scripts, however.

## Training

All training scripts can be found in the training directory.

### Pretraining

Further pretraining your starting checkpoint/model can be done with pretrain.sh, which makes use of Google's provided pretraining scripts.

### Conversion

In order to use the pretrained model with the finetuning scripts, they must first be converted to a pytorch-compatible .bin file.
This can be done with convert_to_torch.sh, which is further documented at https://huggingface.co/transformers/v2.4.0/converting_tensorflow_models.html

### Finetuning

Finetuning your bert model for multi-label classification can be done with finetune.py, with an example of its use in run_finetuning.sh (with 10 random seeds).
In order for this to be used, the labels used will have to be enumerated in a list within finetune.py.

The output will be a .bin model, as well as a pred.tsv file, which contains the predicted labels for each test input.

## K-fold partitioning, evaluation, counting, oversampling, etc.

These tasks have associated scripts, but they may need to be forked/modified for use with data sets other than ESS and Tayside. Some brief descriptions:


- count_most_frequent: used to count and output the most common words in data sets.
- count_tokens: used to count mean, median, max # of tokens per report, to ensure fine-tuning parameters are acceptable.
- count_labels: used to verify # of reports of each label for fine-tuning data sets.
- eval_average: calculates metrics for 10 random initializations, averages them and outputs to tsv.
- oversample: oversamples a dataset using ML-ROS.
- partition_ess_grant: used to partition training set into 5 folds.
