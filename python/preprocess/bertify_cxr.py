from nltk import tokenize
import glob
import sys

# Used to prepare MIMIC-CXR reports for use as pretraining data for BERT.
# That is, the resulting text file will have the form:
# 1 sentence per line.

# Take arg for output file name
output_name = sys.argv[1]

# Load all documents in subfolders recursively.
files = glob.glob('/disk/data2/radiology/users/scassady/data/mimic-cxr/files' + '/**/*.txt', recursive=True)  # Getting what here??

# Create output file.
output = open(output_name, "w+")

# Write all sentences to output file, grouped by document.
for file in files:
	# Get document text from file.
	text = open(file).read().\
		replace('\n', ' ').\
		replace('\r', ' ').\
		replace('FINAL REPORT', '').\
		replace('CHEST (PORTABLE AP)', '').\
		replace('CHEST (PA AND LAT)', '').\
		replace('CHEST RADIOGRAPH', '').\
		replace('PORTABLE', '').\
		replace('CHEST', '').\
		replace('RADIOGRAPH', '').\
		replace('PERFORMED ON', '').\
		replace('DATED', '').\
		replace('CLINICAL HISTORY:', '').\
		replace('TYPE OF', '').\
		replace('REASON FOR EXAMINATION:', '').\
		replace('EXAMINATION:', '').\
		replace('INDICATION:', '').\
		replace('IMPRESSION:', '').\
		replace('PROCEDURE:', '').\
		replace('COMPARISON:', '').\
		replace('COMPARISONS:', '').\
		replace('CLINICAL INFORMATION:', '').\
		replace('STUDY:', '').\
		replace('FINDINGS:', '').\
		replace('TECHNIQUE:', '').\
		replace('HISTORY:', '').\
		replace('HISTORY', '').\
		replace('CLINICAL', '').\
		replace('EXAM:', '').\
		replace('CONCLUSION:', '').\
		replace('FINAL ADDENDUM:', '').\
		replace('REFERENCE EXAM:', '').\
		replace('WET READ:', '').\
		replace('___', '⽥').\
		replace('______________________________________________________________________________','').\
		lstrip()

	# Get sentences from document text.
	sentences = tokenize.sent_tokenize(text)

	# Write individual sentences to output, each on their own line.
	# output.writelines(sentences)
	for sentence in sentences:
		output.write(sentence + "\n")

	# Newline break between documents.
	output.write("\n")

# Close output file.
output.close()
print("DONE")