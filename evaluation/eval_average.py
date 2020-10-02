#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import sys
import statistics
from sklearn.metrics import roc_auc_score
import copy

'''
Calculate Precision, Recall, F1 scores for each label, as well micro and macro values.
This is meant to be used with several different seeds, calculating means and variance for the values.
It is somewhat hardcoded to work with the ESS and Tayside data sets.

Arguments:
1.) GOLD_PATH: gold .csv file.
2.) PRED_DIRECTORY: directory containing the .csv files for each seed.
3.) OUTPUT_PATH: output file name.
4.) DATA_SET: (string) 'ess' or 'tay', which specifies the data set used.
'''

print("-------------------------  reading args...  --------------------------")
GOLD_PATH = sys.argv[1]
PRED_DIRECTORY = sys.argv[2]
OUTPUT_PATH = sys.argv[3]
DATA_SET = sys.argv[4]


# Number of labels. This is needed as the outputting to the final tsv files is hardcoded.
N = 24

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
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.F1 = 0
        self.report_count = 0
        self.positives = 0

    def increment(self, gold_value, pred_value):
        self.report_count += 1
        if gold_value == 1:
            self.positives += 1

        if gold_value == 1 and pred_value == 1:
            self.TP += 1
        elif gold_value == 1 and pred_value == 0:
            self.FN += 1
        elif gold_value == 0 and pred_value == 1:
            self.FP += 1
        elif gold_value == 0 and pred_value == 0:
            self.TN += 1
        else:
            print("**********************************************************INVALID INCREMENT!")
            print("gold_value = ", gold_value)
            print("pred_value = ", pred_value)

    def get_precision(self):
        if (self.TP + self.FP) == 0:
            return 0
        else:
            return self.TP / (self.TP + self.FP)

    def get_recall(self):
        if (self.TP + self.FN) == 0:
            return 0
        else:
            return self.TP / (self.TP + self.FN)

    def get_F1_score(self):
        if (self.get_recall() + self.get_precision()) == 0:
            return 0
        return 2 * (self.get_recall() * self.get_precision()) / (self.get_recall() + self.get_precision())


    # def get_ROC_AUC_score(self):
    #
    #     # y_true = np.array([0, 0, 1, 1])
    #     # y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    #     y_true = np.array([0, 0, 1, 1])
    #     y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    #     return roc_auc_score(y_true, y_scores)
    #
    #     # if (self.get_recall() + self.get_precision()) == 0:
    #     #     return 0
    #     # return 2 * (self.get_recall() * self.get_precision()) / (self.get_recall() + self.get_precision())

def get_ROC_AUC_score(gold_df, pred_df, label):
    y_true = gold_df[label]
    y_scores = pred_df[label]

    # y_true = np.array([0, 0, 1, 1])
    # y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    return roc_auc_score(y_true, y_scores)

label_stat_dict = {}

for label in value_cols:
    label_stat_dict[label] = Label(label)


# ---------------------------- main execution ------------------------------

def process_file(gold_path, pred_path):

    data_dict = {}
    data_dict["Label"] = []
    data_dict["F1"] = []
    data_dict["Precision"] = []
    data_dict["Recall"] = []
    data_dict["ROC-AUC"] = []
    data_dict["TP"] = []
    data_dict["TN"] = []
    data_dict["FP"] = []
    data_dict["FN"] = []
    data_dict["Positives"] = []

    # print("-------------------------  reading args...  --------------------------")
    # GOLD_PATH = sys.argv[1]
    # PRED_PATH = sys.argv[2]
    # OUTPUT_FILE = sys.argv[3]

    # print("-------------------------  loading files...    --------------------------")
    raw_gold_df = pd.read_csv(gold_path,delimiter='\t',encoding='utf-8')
    raw_pred_df = pd.read_csv(pred_path,delimiter='\t',encoding='utf-8')
    # print_dfs(raw_gold_df, raw_pred_df)

    # print("-------------------------  selecting columns...  --------------------------")
    gold_df = raw_gold_df[value_cols]
    pred_df = raw_pred_df[value_cols]
    # print_dfs(gold_df, pred_df)

    # print("-------------------------  thresholding probabilities...   --------------------------")
    # gold_df = gold_df.applymap(lambda x : rnd(x))
    pred_df = pred_df.applymap(lambda x : rnd(x))

    # print("-------------------------  adding ID column back in...   --------------------------")
    gold_df.insert(0, "ID", raw_gold_df["ID"])
    pred_df.insert(0, "ID", raw_gold_df["ID"])
    # print_dfs(gold_df, pred_df)

    # print("-------------------------  tallying negatives, positives ...   --------------------------")

    i = 0
    for ind in gold_df.index:
        i += 1
        for label in value_cols:
            label_stat_dict[label].increment(gold_df[label][ind], pred_df[label][ind])

    # output.write("F1 (ROC_AUC_SCORE)")
    per_label_F1_scores = []
    per_label_ROC_AUC_scores = []

    for i in range(len(value_cols)):
        val = value_cols[i]
        F1 = label_stat_dict[val].get_F1_score()
        per_label_F1_scores.append(F1)

        data_dict["Label"].append(val)
        data_dict["F1"].append(F1)
        data_dict["Recall"].append(label_stat_dict[val].get_recall())
        data_dict["Precision"].append(label_stat_dict[val].get_precision())
        data_dict["TP"].append(label_stat_dict[val].TP)
        data_dict["FP"].append(label_stat_dict[val].FP)
        data_dict["TN"].append(label_stat_dict[val].TN)
        data_dict["FN"].append(label_stat_dict[val].FN)
        data_dict["Positives"].append(label_stat_dict[val].positives)

        try:
            ROC_AUC_Score = get_ROC_AUC_score(pred_df, gold_df, val)
            data_dict["ROC-AUC"].append(ROC_AUC_Score)
        except ValueError:
            data_dict["ROC-AUC"].append(np.nan)

    data_df = pd.DataFrame(data_dict)

    return data_df


is_ess = False

if str(DATA_SET) == "ess":
    is_ess = True
elif str(DATA_SET) == "tay":
    is_ess = False
else:
    raise ValueError("Didn't specify DATA_SET argument properly.")

# OUTPUT_FOLDER = sys.argv[3]

dfs = []
df1 = process_file(GOLD_PATH, PRED_DIRECTORY + "/" + str(41) + "/pred.tsv")
dfs.append(df1)
df1.to_csv(PRED_DIRECTORY + "/metrics_" + str(41) + ".tsv", header=True, sep='\t', index=False)

# process, calculate metrics for each seed's outputs, and append to dataframe list.
for i in range(42, 51):
    df = process_file(GOLD_PATH, PRED_DIRECTORY + "/" + str(i) + "/pred.tsv")

    if is_ess:
        df.loc[13, :] = np.nan
        df.loc[22, :] = np.nan
        df.loc[13,"Label"] = "Tumour, glioma"
        df.loc[22,"Label"] = "TMicrobleed, underspecified"
        # df.drop([13, 22])
    else:
        df.loc[22, :] = np.nan
        df.loc[22,"Label"] = "TMicrobleed, underspecified"
        # df.drop([22])

    dfs.append(df)

    df.to_csv(PRED_DIRECTORY + "/metrics_" + str(i) + ".tsv", header=True, sep='\t', index=False)

# print("dfs[0]:")
# print(dfs[0])
avg_df = copy.deepcopy(dfs[0])
raw = copy.deepcopy(dfs[0])[["F1"]]
prec = copy.deepcopy(dfs[0])[["F1"]]
rec = copy.deepcopy(dfs[0])[["F1"]]
#
# print("raw:")
# print(raw)


f1 = []
micro_F1s = []
macro_F1_means = []
macro_F1_stds = []
macro_precision_means = []
macro_recall_means = []
micro_precisions = []
micro_recalls = []

# average across the processed output of the seeds.
for i in range(0,10):
    j = 41 + i
    s = "F1_" + str(j)
    raw[s] = dfs[i]["F1"]
    s = "Precision_" + str(j)
    prec[s] = dfs[i]["Precision"]
    s = "Recall_" + str(j)
    rec[s] = dfs[i]["Recall"]

    df_label = Label(str(j))
    df_label.TP = dfs[i]["TP"].sum()
    df_label.FP = dfs[i]["FP"].sum()
    df_label.TN = dfs[i]["TN"].sum()
    df_label.FN = dfs[i]["FN"].sum()
    recall = df_label.get_recall()
    precision = df_label.get_precision()
    micro_F1 = df_label.get_F1_score()
    micro_F1s.append(micro_F1)

    macro_F1 = dfs[i]["F1"].mean()
    macro_F1_std = dfs[i]["F1"].std()
    macro_F1_means.append(macro_F1)
    macro_F1_stds.append(macro_F1_std)
    macro_recall_means.append(dfs[i]["Recall"].mean())
    macro_precision_means.append(dfs[i]["Precision"].mean())
    micro_recalls.append(recall)
    micro_precisions.append(precision)


raw.drop(['F1'], axis=1)
prec.drop(['F1'], axis=1)
rec.drop(['F1'], axis=1)

# print("raw:")
# print(raw)

var_F1 = raw.std(axis=1)
var_Prec = prec.std(axis=1)
var_Recall = rec.std(axis=1)

# print("var_F1:")
# print(var_F1)

avg_df["F1_var"] = var_F1
avg_df["Prec_var"] = var_Prec
avg_df["Recall_var"] = var_Recall

cols = ["F1", "Recall", "Precision", "ROC-AUC", "TP", "FP", "TN", "FN", "Positives"]
F1_vals = []
recall_vals = []
precision_vals = []

print_val = True

for col in cols:
    val = 0

    for df in dfs:
        val += df[col]

        if not print_val:
            print(df[col])
            print_val = False


    avg_df[col] = val / 10.0

means = avg_df.mean()

# Place the calculated metrics into the averaged dataframe.
avg_df.at[str(N), "Label"] = "micro_F1_mean"
avg_df.at[str(N),"F1"] = statistics.mean(micro_F1s)
avg_df.at[str(N+1), "Label"] = "micro_F1_std"
avg_df.at[str(N+1),"F1"] = statistics.stdev(micro_F1s)

avg_df.at[str(N+2), "Label"] = "macro_F1_mean"
avg_df.at[str(N+2),"F1"] = statistics.mean(macro_F1_means)
avg_df.at[str(N+3), "Label"] = "macro_F1_std"
avg_df.at[str(N+3),"F1"] = statistics.stdev(macro_F1_means)

avg_df.at[str(N+4), "Label"] = "mean_of_macro_F1_stds"
avg_df.at[str(N+4),"F1"] = statistics.mean(macro_F1_stds)

avg_df.at[str(N+5), "Label"] = "macro_precision mean"
avg_df.at[str(N+5),"F1"] = statistics.mean(macro_precision_means)
avg_df.at[str(N+6), "Label"] = "macro_precision std"
avg_df.at[str(N+6),"F1"] = statistics.stdev(macro_precision_means)

avg_df.at[str(N+7), "Label"] = "micro_precision mean"
avg_df.at[str(N+7),"F1"] = statistics.mean(micro_precisions)
avg_df.at[str(N+8), "Label"] = "micro_precision std"
avg_df.at[str(N+8),"F1"] = statistics.stdev(micro_precisions)

avg_df.at[str(N+10), "Label"] = "macro_recall mean"
avg_df.at[str(N+10),"F1"] = statistics.mean(macro_recall_means)
avg_df.at[str(N+11), "Label"] = "macro_recall std"
avg_df.at[str(N+11),"F1"] = statistics.stdev(macro_recall_means)

avg_df.at[str(N+12), "Label"] = "micro_recalls mean"
avg_df.at[str(N+12),"F1"] = statistics.mean(micro_recalls)
avg_df.at[str(N+13), "Label"] = "micro_recall std"
avg_df.at[str(N+13),"F1"] = statistics.stdev(micro_recalls)

# The original code used, in case the above doesn't work.
# avg_df.at["24", "Label"] = "micro_F1_mean"
# avg_df.at["24","F1"] = statistics.mean(micro_F1s)
# avg_df.at["25", "Label"] = "micro_F1_std"
# avg_df.at["25","F1"] = statistics.stdev(micro_F1s)
#
# avg_df.at["26", "Label"] = "macro_F1_mean"
# avg_df.at["26","F1"] = statistics.mean(macro_F1_means)
# avg_df.at["27", "Label"] = "macro_F1_std"
# avg_df.at["27","F1"] = statistics.stdev(macro_F1_means)
#
# avg_df.at["28", "Label"] = "mean_of_macro_F1_stds"
# avg_df.at["28","F1"] = statistics.mean(macro_F1_stds)
#
# avg_df.at["29", "Label"] = "macro_precision mean"
# avg_df.at["29","F1"] = statistics.mean(macro_precision_means)
# avg_df.at["30", "Label"] = "macro_precision std"
# avg_df.at["30","F1"] = statistics.stdev(macro_precision_means)
#
# avg_df.at["31", "Label"] = "micro_precision mean"
# avg_df.at["31","F1"] = statistics.mean(micro_precisions)
# avg_df.at["32", "Label"] = "micro_precision std"
# avg_df.at["32","F1"] = statistics.stdev(micro_precisions)
#
# avg_df.at["33", "Label"] = "macro_recall mean"
# avg_df.at["33","F1"] = statistics.mean(macro_recall_means)
# avg_df.at["34", "Label"] = "macro_recall std"
# avg_df.at["34","F1"] = statistics.stdev(macro_recall_means)
#
# avg_df.at["35", "Label"] = "micro_recalls mean"
# avg_df.at["35","F1"] = statistics.mean(micro_recalls)
# avg_df.at["36", "Label"] = "micro_recall std"
# avg_df.at["36","F1"] = statistics.stdev(micro_recalls)

# print("avg_df:")
# print(avg_df)

# Output to csv
avg_df.to_csv(OUTPUT_PATH, header=True, sep='\t', index=False)