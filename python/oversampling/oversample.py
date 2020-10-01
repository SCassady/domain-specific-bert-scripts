
import pandas as pd
import sys
import copy
import statistics
import random
import psutil

all_labels = ["Ischaemic stroke, deep, recent",
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

nonzero_labels = []



# ---------------------------- main execution ------------------------------
print("-------------------------  reading args...  --------------------------")
GOLD_PATH = sys.argv[1]
OUTPUT_FILE = sys.argv[2]
PROPORTION_TO_CLONE = float(sys.argv[3])
MEAN_DIVISION_FACTOR = float(sys.argv[4])

input_df = pd.read_csv(GOLD_PATH, delimiter='\t', encoding='utf-8')
# gold_df = raw_gold_df[label_names]
output_df = copy.deepcopy(input_df)


# -------------------------------CREATE DATA STRUCTURES ----------------------------
report_dict = {}

for label in all_labels:
    report_dict[label] = []

for index, row in input_df.iterrows():
    for label in all_labels:
        if row[label] == 1:
            report_dict[label].append(row)
    # print(row['c1'], row['c2'])

for label in all_labels:
    if len(report_dict[label]) > 0:
        nonzero_labels.append(label)


# --------------------------------PERFORM SAMPLING---------------------------
def print_label_counts():
    print("---------------------LABEL COUNTS:")

    for l in nonzero_labels:
        print(l + " count: " + str(len(report_dict[l])))

def get_imbalance_ratio(selected):
    # Get most common label, and it's count
    # biggest_label = label_names[0]
    highest_count = -1

    for l in nonzero_labels:
        if len(report_dict[l]) > highest_count:
            highest_count = len(report_dict[l])
            # biggest_label = l

    # Get ratio
    label_count = len(report_dict[selected])

    return highest_count / label_count

def calculate_mean_IR():
    ir_values = []

    for l in nonzero_labels:
        ir_values.append(get_imbalance_ratio(l))

    return statistics.mean(ir_values)

def randomly_clone_report_of_label(l):
    # print("len(report_dict[l]):", len(report_dict[l]))
    roll = random.randint(0, len(report_dict[l])-1)
    # print("roll:", roll)
    report = report_dict[l][roll]
    clone = copy.deepcopy(report)
    report_dict[l].append(clone)
    # df = df.append(clone)
    return clone

    # print(report)

def ML_ROS(samples_to_clone):
    # samples_to_clone = int(len(reports) * proportion_to_clone)
    cloned_reports = []

    mean_IR = calculate_mean_IR() / MEAN_DIVISION_FACTOR
    print("mean_IR: ", mean_IR)

    min_bags = []
    for l in nonzero_labels:
        ratio = get_imbalance_ratio(l)
        if ratio > mean_IR:
            min_bags.append(l)
            print("added to min_bags: ", l)
            print("(ratio): ", ratio)

    print("cloning reports:")
    while samples_to_clone > 0:
        for bag in min_bags:
            print("samples_to_clone:", samples_to_clone)
            # print("% memory available:", psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
            # clone a sample in bag
            cloned_reports.append(randomly_clone_report_of_label(bag))

            bag_IR = get_imbalance_ratio(bag)
            if bag_IR <= mean_IR:
                min_bags.remove(bag)
                print("bag removed: ", bag)
                print("(bag_IR): ", bag_IR)

            samples_to_clone -= 1

            if samples_to_clone < 0:
                break

        if len(min_bags) == 0:
            print("min_bags empty; breaking.")
            break

    return cloned_reports


# _------------------------------PRINT FINAL DISTRIBUTION------------------------------

# PROPORTION_TO_CLONE = 0.25
num_to_clone = int(PROPORTION_TO_CLONE * input_df.shape[0])
print("num_to_clone:", num_to_clone)
print_label_counts()
clones = ML_ROS(num_to_clone)
print_label_counts()

for clone in clones:
    output_df = output_df.append(clone)

print("input_df.shape: ", input_df.shape)
print("output_df.shape: ", output_df.shape)
# _------------------------------SAVE OUTPUT TO CSV------------------------------
output_df.to_csv(OUTPUT_FILE, header=True, sep='\t', index=False)
print("--------------COMPLETED----------------------")
