from nltk import tokenize
import glob
import sys
import re
from lxml import etree

# Used to prepare Tayside reports for use as pretraining data for BERT.
# That is, the resulting text file will have the form:
# 1 sentence per line.

# Take arg for output file name
output_name = sys.argv[1]

# Load all documents in subfolders recursively.
# files = glob.glob('/disk/data2/radiology/users/scassady/data/mimic-cxr/files' + '/**/*.txt', recursive=True)  # Getting what here??
file = '/disk/data2/radiology/users/scassady/data/tayside/orig/Project 3271 - Tayside Scan Reports_v3.txt'
devIndexFile = 'data/tayside/gold/dev/taydevall-documentIDs'
testIndexFile = 'data/tayside/gold/test/taytestall-documentIDs'
dev_input_folder = "/disk/data2/radiology/users/scassady/data/tayside/newgold/dev/taydevall-Will/"
test_input_folder = "/disk/data2/radiology/users/scassady/data/tayside/newgold/test/taytestall-Will/"

# ----------------------------------------------------------------

# EXAMPLE:
# <?xml version="1.0" encoding="UTF-8"?>
# <document version="5" id="0014">
# <meta>
# <attr name="origid">17</attr>                 <--------- HERE!
# <attr name="scantype">CT</attr>
# </meta>
# <text>
# <p><s><w>Report 0014: CSKUH (CT)</w></s></p>

def getIndices(files):
	output = []
	for file in files:
		tree = etree.parse(file)
		tagged = etree.tostring(tree, encoding='utf8', method='xml')
		root = etree.fromstring(tagged)

		origID_node = root.xpath("//attr[@name='origid']")[0]
		origID = int(''.join(origID_node.itertext()))
		output.append(origID)
		print("file: ", file)
		print("origID: ", origID)
	return output

allIntIndices = []

dev_files = glob.glob(dev_input_folder + '/*.xml', recursive=True)
test_files = glob.glob(dev_input_folder + '/*.xml', recursive=True)

print("len(dev_files): ", len(dev_files))
print("len(test_files): ", len(test_files))

devIndices = getIndices(dev_files)
testIndices = getIndices(test_files)
allIntIndices = devIndices + testIndices

# Create output file.
output = open(output_name, "w+")

# Write all sentences to output file, grouped by document.
# Get document text from file.
text = open(file).read()

# docs = text.split("|~")
docs = re.split('[0-9]+\|',text)
docs.pop(0)
print("len(docs): ", len(docs))
print("len(allIntIndices): ", len(allIntIndices))
print("allIntIndices: ", allIntIndices)
n = 0

for i in range(len(docs)):
	if i+1 not in allIntIndices:
		n += 1
		# Get sentences from document text.
		sentences = tokenize.sent_tokenize(docs[i])
		# sentences.pop()

		# Write individual sentences to output, each on their own line.
		# output.writelines(sentences)
		for sentence in sentences:
			output.write(sentence + "\n")

		# Newline break between documents.
		output.write("\n")
	if (i+1) == 17 or (i+1) == 14:
		print("i: ", i)
		print("doc: ", docs[i])

print("docs kept: ", n)

# Close output file.
output.close()
print("DONE")