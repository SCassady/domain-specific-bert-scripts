#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import sys


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

def print_dfs(df1,df2):
    print("~~~~~~ print_dfs() called ~~~~~~~")
    print("gold_df.shape: ", df1.shape)
    print("pred_df.shape: ", df2.shape)

class Label():
    def __init__(self, label_type):
        self.label_type = label_type
        self.report_count = 0
        self.positives = 0

    def increment(self, gold_value):
        self.report_count += 1
        if gold_value == 1:
            self.positives += 1

label_stat_dict = {}

for label in value_cols:
    label_stat_dict[label] = Label(label)


# ---------------------------- main execution ------------------------------

print("-------------------------  reading args...  --------------------------")
GOLD_PATH = sys.argv[1]

print("-------------------------  loading files...    --------------------------")
raw_gold_df = pd.read_csv(GOLD_PATH,delimiter='\t',encoding='utf-8')

print("-------------------------  selecting columns...  --------------------------")
gold_df = raw_gold_df[value_cols]

# print("gold_df.to_numpy().sum(): ", gold_df.to_numpy().sum())
# print("pred_df.to_numpy().sum(): ", pred_df.to_numpy().sum())


print("-------------------------  adding ID column back in...   --------------------------")
gold_df.insert(0, "ID", raw_gold_df["ID"])


print("-------------------------  tallying ...   --------------------------")

i = 0
for ind in gold_df.index:
    i += 1
    for label in value_cols:
        label_stat_dict[label].increment(gold_df[label][ind])


print("~~~~~~~~~~~~~~~~~~~~~~  Final counts : ~~~~~~~~~~~~~~~~~~~~~~s")
for label in value_cols:
    count_str = str(label) + ": " + str(label_stat_dict[label].positives)
    print(count_str)


print(">>>>>>>>>>>> TOTAL REPORT COUNT = ", i)
