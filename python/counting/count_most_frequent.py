#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import sys
import nltk
import datetime

def rnd(x):
    if pd.to_numeric(x) >= 0.5:
        return 1
    else:
        return 0


# print("----------------------------------------------------------------------------Word token lengths:")
stopwords = nltk.corpus.stopwords.words('english')


def get_most_frequent_words(text, N, title):
    print("")
    print("---> getting most frequent words for:",title)
    print("> start: ", datetime.datetime.now().time())
    words = nltk.tokenize.word_tokenize(text)
    print("> tokenizing done: ", datetime.datetime.now().time())
    print("num_words:",len(words))
    word_dist = nltk.FreqDist(w.lower() for w in words if (w.lower() not in stopwords and w.lower().isalpha()))
    print("> FreqDist done: ", datetime.datetime.now().time())
    most_common_words = word_dist.most_common(N)
    print("most common words:")
    print(most_common_words)
    print(">> end: ", datetime.datetime.now().time())

    output = open(title + "_" + str(N) + "_most_freq_words.txt", "w+")

    for word in most_common_words:
        output.write(str(word))
    output.close()

    return most_common_words


print("loading data...")

III_PATH = "/disk/data2/radiology/users/scassady/data/mimiciii/NOTEEVENTS.csv"

ESS_TRAIN_PATH = "/disk/data2/radiology/users/scassady/data/finetuning/ess_grant-merge/train.tsv"
ESS_TEST_PATH = "/disk/data2/radiology/users/scassady/data/finetuning/ess_grant-merge/test.tsv"
TAY_TRAIN_PATH = "/disk/data2/radiology/users/scassady/data/finetuning/tay-merge/train.tsv"
TAY_TEST_PATH = "/disk/data2/radiology/users/scassady/data/finetuning/tay-merge/test.tsv"

TAY_UNLABELLED_PATH = "/disk/data2/radiology/users/scassady/data/processed/tayside.txt"
CXR_PATH = "/disk/data2/radiology/users/scassady/data/processed/mimic-cxr.txt"

NUM_WORDS = 100
print("reading data...")

ess_train_df = pd.read_csv(ESS_TRAIN_PATH,delimiter='\t',encoding='utf-8')
ess_test_df = pd.read_csv(ESS_TEST_PATH,delimiter='\t',encoding='utf-8')
tay_train_df = pd.read_csv(TAY_TRAIN_PATH,delimiter='\t',encoding='utf-8')
tay_test_df = pd.read_csv(TAY_TEST_PATH,delimiter='\t',encoding='utf-8')


ess_text = ' '.join(ess_train_df["TEXT"]) + ' ' + ' '.join(ess_test_df["TEXT"])
tay_text = ' '.join(tay_train_df["TEXT"]) + ' ' + ' '.join(tay_test_df["TEXT"])


with open(CXR_PATH, 'r') as file:
    cxr_text = file.read().replace('\n', '')
with open(TAY_UNLABELLED_PATH, 'r') as file:
    tay_un_text = file.read().replace('\n', '')

# ----------------------------------------------------------------III
iii_df = pd.read_csv(III_PATH,encoding='utf-8')
iii_text = ' '.join(iii_df["TEXT"])
print("iii_text count: ", len(iii_text))
# print("iii_text type: ", type(iii_text))
# print(iii_text[0:5000])

print("counting...")
print("ess_text count: ", len(ess_text))
print("tay_text count: ", len(tay_text))
print("cxr_text count: ", len(cxr_text))
print("tay_un_text count: ", len(tay_un_text))

print("getting frequencies------------")
ess_most_common = get_most_frequent_words(ess_text, NUM_WORDS, "ess")
tay_most_common = get_most_frequent_words(tay_text, NUM_WORDS, "tay")
cxr_most_common = get_most_frequent_words(cxr_text, NUM_WORDS, "cxr")
tay_un_most_common = get_most_frequent_words(tay_un_text, NUM_WORDS, "tay_unlabelled")
iii_most_common = get_most_frequent_words(iii_text, NUM_WORDS, "iii")


output_df = pd.DataFrame(
    {
        'ess_words': [item[0] for item in ess_most_common],
        'ess_counts': [item[1] for item in ess_most_common],
        'tay_words': [item[0] for item in tay_most_common],
        'tay_counts': [item[1] for item in tay_most_common],
        'cxr_words': [item[0] for item in cxr_most_common],
        'cxr_counts': [item[1] for item in cxr_most_common],
        'tay_un_words': [item[0] for item in tay_un_most_common],
        'tay_un_counts': [item[1] for item in tay_un_most_common],
        'iii_words': [item[0] for item in iii_most_common],
        'iii_counts': [item[1] for item in iii_most_common],
    })

OUTPUT_PATH = sys.argv[1]

print(output_df)
output_df.to_csv("/disk/data2/radiology/users/scassady/data/" + OUTPUT_PATH, header=True, sep='\t', index=False)
# output = open("100_words.txt", "w+")
# output.write("")
# output.close()
