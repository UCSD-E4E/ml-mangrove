from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
import os
import shutil
import sys
import time

#load label file for class names
def load_labels(label_file):
	label = []
	proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
	for l in proto_as_ascii_lines:
		label.append(l.rstrip())
	return label

#load trained neural network
def load_graph(model_file):
	graph = tf.Graph()
	graph_def = tf.GraphDef()
	with open(model_file, "rb") as f:
		graph_def.ParseFromString(f.read())
	with graph.as_default():
		tf.import_graph_def(graph_def)
	return graph

#make directories for each output labeled image
def mkdir(labels, output_dir):
	for line in labels:
		final_directory = os.path.join(output_dir, line)
		if not os.path.exists(final_directory):
			os.makedirs(final_directory)
			print("Successfully made directory at" + final_directory)
		else:
			print("Directory with name " + line + " already exists!")
			sys.exit()

#movefiles to specified output directory with subsequent class folders
def movefiles(files,classes,output_dir):
        names = []
	for file in files:
		names.append(os.path.basename(file))
	for i in range(len(files)):
                file = str(files[i])
                print(file)
		shutil.move(file[:-1], os.path.join(str(output_dir) + "/" + str(classes[i]), str(names[i]))) 

#get top percentages of classified image
def get_topk(result, n):
	topk = result[1::n+1]
	files = result[0::n+1]
	classes = []
	for i in range(len(topk)):
		classes.append(topk[i].split(None,1)[0])
	return files, classes

#find files in specified image directory
def find_files(image_directory):
	file_list = []
	for root, dirs, files in os.walk(os.path.abspath(image_directory)):
   		for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
	input_name = "file_reader"
	output_name = "normalized"
	file_reader = tf.read_file(file_name, input_name)
	if file_name.endswith(".png"):
		image_reader = tf.image.decode_png(file_reader, channels=3, name="png_reader")
	elif file_name.endswith(".gif"):
		image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name="gif_reader"))
	elif file_name.endswith(".bmp"):
		image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
	else:
		image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
	float_caster = tf.cast(image_reader, tf.float32)
	dims_expander = tf.expand_dims(float_caster, 0)
	resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
	normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
	sess = tf.Session()
	result = sess.run(normalized)
	return result


if __name__ == "__main__":
	#default args
	file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
	model_file = \
	"tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
	label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
	input_height = 299
	input_width = 299
	input_mean = 0
	input_std = 255
	input_layer = "input"
	output_layer = "InceptionV3/Predictions/Reshape_1"
	#add arguments for classifier
	parser = argparse.ArgumentParser()
	parser.add_argument("--images", help="image directory to be processed")
	parser.add_argument("--graph", help="graph/model to be executed")
	parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--output_dir", help="location of where you want to move images to")
	parser.add_argument("--input_height", type=int, help="input height")
	parser.add_argument("--input_width", type=int, help="input width")
	parser.add_argument("--input_mean", type=int, help="input mean")
	parser.add_argument("--input_std", type=int, help="input std")
	parser.add_argument("--input_layer", help="name of input layer")
	parser.add_argument("--output_layer", help="name of output layer")
	args = parser.parse_args()

	if args.graph:
		model_file = args.graph
	if args.images:
		image_directory = args.images
	if args.labels:
		label_file = args.labels
	if args.input_height:
		input_height = args.input_height
	if args.input_width:
		input_width = args.input_width
	if args.input_mean:
		input_mean = args.input_mean
	if args.input_std:
		input_std = args.input_std
	if args.input_layer:
		input_layer = args.input_layer
	if args.output_layer:
		output_layer = args.output_layer

	graph = load_graph(model_file)
    classes = load_labels(label_file)
	results = []
    file_list = find_files(image_directory)
    total_images = 0

	#process imagery
    t1 = time.time()
	for file in file_list:
		if file.endswith(".jpg"):
			t = read_tensor_from_image_file(
				file,
				input_height=input_height,
				input_width=input_width,
				input_mean=input_mean,
				input_std=input_std)
			input_name = "import/" + input_layer
			output_name = "import/" + output_layer
			input_operation = graph.get_operation_by_name(input_name)
			output_operation = graph.get_operation_by_name(output_name)

			with tf.Session(graph=graph) as sess:
				results = sess.run(output_operation.outputs[0], {
				input_operation.outputs[0]: t})
			results = np.squeeze(results)

			top_k = results.argsort()[-5:][::-1]
			labels = load_labels(label_file)
			for i in top_k:
				print(labels[i], results[i])
				results.append(classes[i] + " " + str(results[i]) + "\n")
        total_images = total_images + 1
    t2 = time.time()
    total_time = t2 - t1
    
    #move classified images into folders
	label_classes = open(label_file, "r")
	files, classes  = get_topk(results,len(classes))
	mkdir(label_classes, output)
	movefiles(files, classes, output)
    print("This entire classification (in seconds) took: " + str(total_time) + " seconds.")
    print("Average classification time per image (in seconds) was " + str(total_time / total_images) + " seconds.")