### BERT domain-specific training scripts

Scripts to further pretrain pretrained bert models on more data, and then to finetune on a task. Also includes evaluation scripts and counting scripts, but some are currently hardcoded for the ESS and Tayide datasets.

In order to use this, the following should also be installed:
- the official bert repository: https://github.com/google-research/bert
- a pretrained bert-base model; for example, BlueBERT-Base, Uncased, PubMed+MIMIC-III

For the MSc work, 2 environments were used:

for pretraning, with the following installed:
- python                    3.6.10  
- bert-tensorflow           1.0.1  
- tensorflow                1.11.0        
- tensorflow-base           1.11.0        
- tensorflow-gpu            1.11.0     
- lxml                      4.5.1 (for creating tsv files)
- nltk                      3.5 (if counting tokens)

for converting the tensorflow model to pytorch, and fine-tuning:
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

### Training

All training scripts can be found in the training directory.

#### Pretraining

Further pretraining your starting checkpoint/model can be done with pretrain.sh, which makes use of Google's provided pretraining scripts.

#### Conversion

In order to use the pretrained model with the finetuning scripts, they must first be converted to a pytorch-compatible .bin file.
This can be done with convert_to_torch.sh, which is further documented at https://huggingface.co/transformers/v2.4.0/converting_tensorflow_models.html

#### Finetuning

Finetuning your bert model for multi-label classification can be done with finetune.py, with an example of its use in run_finetuning.sh (with 10 random seeds).
In order for this to be used, the labels used will have to be enumerated in a list within finetune.py.

The output will be a .bin model, as well as a pred.tsv file, which contains the predicted labels for each test input.

### Evaluation, counting, oversampling

These tasks have associated scripts, but they may need to be forked/modified for use with data sets other than ESS and Tayside.

















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