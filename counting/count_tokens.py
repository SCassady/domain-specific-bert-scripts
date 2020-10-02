#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import sys
import statistics
from nltk import tokenize

gold_cols = ["ID",
             "Ischaemic stroke, deep, recent",
            "Ischaemic stroke, deep, old",
            "Ischaemic stroke, cortical, recent",
            "Ischaemic stroke, cortical, old",
            "Ischaemic stroke, underspecified",
            "Haemorrhagic stroke, deep, recent",
            "Haemorrhagic stroke, deep, old",
            "Haemorrhagic stroke, lobar, recent",
            "Haemorrhagic stroke, lobar, old",
            "Haemorrhagic stroke, underspecified",
            "Stroke, underspecified",
            "Tumour, meningioma",
            "Tumour, metastasis",
            "Tumour, glioma",
            "Tumour, other",
            "Small vessel disease",
            "Atrophy",
            "Subdural haematoma",
            "Subarachnoid haemorrhage, aneurysmal",
            "Subarachnoid haemorrhage, other",
            "Microbleed, deep",
            "Microbleed, lobar",
            "Microbleed, underspecified",
            "Haemorrhagic transformation"]

pred_cols = ["id",
             "Ischaemic stroke, deep, recent",
            "Ischaemic stroke, deep, old",
            "Ischaemic stroke, cortical, recent",
            "Ischaemic stroke, cortical, old",
            "Ischaemic stroke, underspecified",
            "Haemorrhagic stroke, deep, recent",
            "Haemorrhagic stroke, deep, old",
            "Haemorrhagic stroke, lobar, recent",
            "Haemorrhagic stroke, lobar, old",
            "Haemorrhagic stroke, underspecified",
            "Stroke, underspecified",
            "Tumour, meningioma",
            "Tumour, metastasis",
            "Tumour, glioma",
            "Tumour, other",
            "Small vessel disease",
            "Atrophy",
            "Subdural haematoma",
            "Subarachnoid haemorrhage, aneurysmal",
            "Subarachnoid haemorrhage, other",
            "Microbleed, deep",
            "Microbleed, lobar",
            "Microbleed, underspecified",
            "Haemorrhagic transformation"]

gorinski_labels = ["Ischaemic stroke",
                    "Haemorrhagic stroke",
                    "location:cortical",
                    "location:deep",
                    "time:old",
                    "time:recent",
                    "Subarachnoid haemorrhage",
                    "Microbleed"]

value_cols = ["Ischaemic stroke, deep, recent",
            "Ischaemic stroke, deep, old",
            "Ischaemic stroke, cortical, recent",
            "Ischaemic stroke, cortical, old",
            "Ischaemic stroke, underspecified",
            "Haemorrhagic stroke, deep, recent",
            "Haemorrhagic stroke, deep, old",
            "Haemorrhagic stroke, lobar, recent",
            "Haemorrhagic stroke, lobar, old",
            "Haemorrhagic stroke, underspecified",
            "Stroke, underspecified",
            "Tumour, meningioma",
            "Tumour, metastasis",
            "Tumour, glioma",
            "Tumour, other",
            "Small vessel disease",
            "Atrophy",
            "Subdural haematoma",
            "Subarachnoid haemorrhage, aneurysmal",
            "Subarachnoid haemorrhage, other",
            "Microbleed, deep",
            "Microbleed, lobar",
            "Microbleed, underspecified",
            "Haemorrhagic transformation"]

def rnd(x):
    if pd.to_numeric(x) >= 0.5:
        return 1
    else:
        return 0


# ---------------------------- main execution ------------------------------

print("-------------------------  reading args...  --------------------------")
GOLD_PATH = sys.argv[1]
# PRED_PATH = sys.argv[2]
# OUTPUT_FILE = sys.argv[3]

print("-------------------------  loading files...    --------------------------")
raw_gold_df = pd.read_csv(GOLD_PATH,delimiter='\t',encoding='utf-8')
print(raw_gold_df.head())

print("----------------------------------------------------------------------------Word token lengths:")

# i = 0
token_lengths = []

for ind in raw_gold_df.index:
    # i += 1
    text = raw_gold_df["TEXT"][ind]
    word_tokens = tokenize.word_tokenize(text)
    tok_count = len(word_tokens)
    print(tok_count)
    token_lengths.append(tok_count)

print("mean: ", str(statistics.mean(token_lengths)))
print("median: ", str(statistics.median(token_lengths)))
print("median: ", str(statistics.median(token_lengths)))
print("max value: ", str(max(token_lengths)))