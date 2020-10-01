from lxml import etree
import csv
import glob
import random
import copy

# Used to partition ESS-grant into 5 folds, for k-fold cross-validation.

label_names = ["Ischaemic stroke, deep, recent",
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


def clean_string(input):
    return " ".join(input.split())

def get_labels(yes_nodes):
    # Get a list representing whether or not the file has been tagged with each label.
    labels = []
    for i in range(24):
        labels.append(0)

    if len(yes_nodes) > 0:
        for node in yes_nodes:
            node_text = ''.join(node.itertext())

            for j in range(len(label_names)):
                if label_names[j] in node_text:
                    labels[j] = 1

    return labels

def process_files(input_files, output_file, ID_prefix):
    with open(output_file, 'w') as tsvfile:
        tsv_writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')

        # Write column names
        tsv_writer.writerow(["ID",
                             "TEXT",
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
                            "Haemorrhagic transformation"])

        # Write labels and text to rows
        conclusion_count = 0
        no_conclusion_count = 0
        no_text_count = 0

        for file in input_files:
            # print("\nfile: ", file)
            row = []

            tree = etree.parse(file)
            tagged = etree.tostring(tree, encoding='utf8', method='xml')
            root = etree.fromstring(tagged)
            p_nodes = root.xpath("//p")
            print("len(p_nodes): ", len(p_nodes))

            doc_ID = root.xpath("//document[@id]/@id")[0]
            row.append(ID_prefix + doc_ID)

            if len(p_nodes) > 28:
                print("p_nodes[25] text: ", ''.join(p_nodes[25].itertext()))
                print("p_nodes[27] text: ", ''.join(p_nodes[27].itertext()))
                conclusion_count += 1
                report_node = p_nodes[26]
                full_text = ''.join(report_node.itertext())
                conclusion_node = p_nodes[28]
                conclusion_text = ''.join(conclusion_node.itertext())
                full_text = full_text + conclusion_text
                full_text = clean_string(full_text)

                # row.append("lorem ipsum")
                row.append(full_text)

                yes_nodes = root.xpath("//p[@label='yes']")
                label_outputs = get_labels(yes_nodes)
                row += label_outputs
                tsv_writer.writerow(row)

            elif len(p_nodes) > 26:
                print("p_nodes[25] text: ", ''.join(p_nodes[25].itertext()))
                no_conclusion_count += 1
                report_node = p_nodes[26]
                full_text = ''.join(report_node.itertext())
                full_text = clean_string(full_text)

                # row.append("lorem ipsum")
                row.append(full_text)

                yes_nodes = root.xpath("//p[@label='yes']")
                label_outputs = get_labels(yes_nodes)
                row += label_outputs
                tsv_writer.writerow(row)

            else:
                no_text_count += 1
                print("file didn't have any text: ", file)

        total_successes = conclusion_count + no_conclusion_count
        print("conclusion_count: ", conclusion_count)
        print("no_conclusion_count: ", no_conclusion_count)
        print("total successes count: ", total_successes)
        print("no_text_count: ", no_text_count)
        print("COMPLETED")

random.seed(42)

dev_input_folder = "/disk/data2/radiology/users/scassady/data/ess/gold/devall/"
test_input_folder = "/disk/data2/radiology/users/scassady/data/ess/gold/testall-Grant/"

N = 5

dev_files = glob.glob(dev_input_folder + '/*.xml', recursive=True)
random.shuffle(dev_files)
# eval_count = int(0.2 * len(dev_files))

eval_count = int((1.0/N) * len(dev_files))
print("N", N)
print("len(dev_files)", len(dev_files))
print("eval_count", eval_count)
partitions = []
partitions.append(dev_files[0:eval_count])

for i in range(1, N-1):
    partitions.append(dev_files[i*eval_count:(i+1)*eval_count])

partitions.append(dev_files[(N-1)*eval_count:len(dev_files)])
print("len(partitions)", len(partitions))

for i in range(N):
    temp_partitions = copy.copy(partitions)
    eval_files = temp_partitions.pop(i)

    train_files = []

    for l in temp_partitions:
        train_files += l

    print("len(train_files):", len(train_files))
    print("len(eval_files):", len(eval_files))

    print("------> Building TRAIN file ----------------------------------------------------")
    train_output_file = '/disk/data2/radiology/users/scassady/data/finetuning/ess_grant-p5/p' + str(i) + '/train.tsv'
    process_files(train_files, train_output_file, "TR")
    print("------> Building EVAL file ----------------------------------------------------")
    eval_output_file = '/disk/data2/radiology/users/scassady/data/finetuning/ess_grant-p5/p' + str(i) + '/dev.tsv'
    process_files(eval_files, eval_output_file, "EV")