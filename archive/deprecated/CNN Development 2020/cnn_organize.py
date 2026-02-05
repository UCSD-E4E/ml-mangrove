import argparse
import os
import tensorflow as tf
import shutil
import sys

def load_labels(label_file):
	label = []
	proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
	for l in proto_as_ascii_lines:
		label.append(l.rstrip())
	return label

def get_topk(result_file, n):
	file = open(result_file,"r")
	lines = file.readlines()
	topk = lines[1::n+1]
	files = lines[0::n+1]
	classes = []
	for i in range(len(topk)):
		classes.append(topk[i].split(None,1)[0])
	return files, classes

def mkdir(labels, output_dir):
	for line in labels:
		final_directory = os.path.join(output_dir, line.strip())
		if not os.path.exists(final_directory):
			os.makedirs(final_directory)
			print("Successfully made directory at" + final_directory)
		else:
			print("Directory with name " + line + " already exists!")
			sys.exit()

def movefiles(files,classes,output_dir):
	names = []
	count = 0
	for file in files:
		names.append(os.path.basename(file))
	for i in range(len(files)):
		file = str(files[i])
		print(file)
		shutil.move(file[:-1], os.path.join(str(output_dir) + "/" + str(classes[i]), str(names[i])))
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--labels", help="location of label file")
	parser.add_argument("--output_dir", help="location of where you want to move images to")
	parser.add_argument("--result_file", help="location of the result file from annotate.py")
	args = parser.parse_args()

	if args.labels:
		label_file = args.labels
	if args.result_file:
		result = args.result_file
	if args.output_dir:
		output = args.output_dir 


	classes = load_labels(label_file)
	n = len(classes)
	label_classes = open(label_file, "r")
	files, classes  = get_topk(result,n)
	mkdir(label_classes, output)
	movefiles(files, classes, output)

	

