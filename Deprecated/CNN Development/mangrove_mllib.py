from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image

import shutil
import numpy as np
import tensorflow as tf
import argparse
import os

#The following functions are taken from the TensorFlow library
def load_labels(label_file):
	label = []
	proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
	for l in proto_as_ascii_lines:
		label.append(l.rstrip())
	return label

def load_graph(model_file):
	graph = tf.Graph()
	graph_def = tf.GraphDef()
	with open(model_file, "rb") as f:
		graph_def.ParseFromString(f.read())
	with graph.as_default():
		tf.import_graph_def(graph_def)
	return graph

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

'''
function: tensor_classify_tile

Summary: Takes an input image, previously trained model file along with the label file to classify the image. Adjust the hyperparameters below as 
necessary 

inputs: file_name = any location of a .jpg file that you want to classify
        model_file = any location of the model file that you previously trained for that came from cnn_retrain.py
        label_file = the label file that also came from the training from cnn_retrain.py
returns: a classification for the single input image with the schema
		|_class1 prob
		|_class2 prob
		|_class3 prob
'''

def tensor_classify_tile(file_name, model_file, label_file):
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Placeholder"
    output_layer = "final_result"
    graph = load_graph(model_file)
    t = read_tensor_from_image_file(
        file_name,
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
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
        print(labels[i], results[i])
    return top_k
'''
function: tensor_classify_directory

Summary: Takes an input directory of .jpg images, previously trained model file along with the label file to classify the image. Adjust the hyperparameters below as 
necessary 

inputs: file_name = any directory with .jpg files that you want to classify
        model_file = any location of the model file that you previously trained for that came from cnn_retrain.py
        label_file = the label file that also came from the training from cnn_retrain.py
returns: a list of classifications for each file in the directory with the schema
		|_file
		|_class1 prob
		|_class2 prob
		|_class3 prob
		|_file2
		...
'''
def tensor_classify_directory(image_directory, model_file, label_file):
	input_height = 299
	input_width = 299
	input_mean = 0
	input_std = 255
	input_layer = "Placeholder"
	output_layer = "final_result"
	graph = load_graph(model_file)
	results = []
	file_list = []
	for root, dirs, files in os.walk(os.path.abspath(image_directory)):
   		for file in files:
                        file_list.append(os.path.join(root, file))
	#process imagery
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
			results.append(file)
			for i in top_k:
				print(labels[i], results[i])
				results.append(labels[i] + " " + str(results[i]))
	return results


'''
function: seg_classify_tile

Summary: Takes already segmented images from orfeotoolbox and calculates the percentages of each pixel for mangrove species

inputs: file_directory = any directory with images classified tiles using orfeo toolbox with the colormap as defined below 

returns: returns a list of classification with the schema
		|_class1 prob
		|_class2 prob
		|_class3 prob

'''
def seg_classify_tile(image_directory):
	color_red = (255,0,0)
	color_black = (0,0,0)
	color_white = (255,255,255)
	file_list = []
	for root, dirs, files in os.walk(os.path.abspath(image_directory)):
		for file in files:
			file_list.append(os.path.join(root, file))         
	for file in file_list:
		im = Image.open(file)
		red_count = black_count = white_count = 0
		for pixel in im.getdata():
			if pixel == color_red:
				red_count += 1
			elif pixel == color_black:
				black_count += 1
			elif pixel ==  color_white:
				white_count += 1
		percentage_red = float(red_count) / float(65536)
		percentage_black = float(black_count) / float(65536)
		percentage_white = float(black_count) / float(65536)
		return[["Red", percentage_red], ["Black" , percentage_black],["White", percentage_white]]

'''
function: mkdir

Summary: creates direcories for each class in the label_file 

inputs: labels = label file created from the tensor training process
        output_dir = where you want to put the directories of the labels
returns: void

'''
def mkdir(labels, output_dir):
	for line in labels:
		final_directory = os.path.join(output_dir, line)
		if not os.path.exists(final_directory):
			os.makedirs(final_directory)
			print("Successfully made directory at" + final_directory)
		else:
			print("Directory with name " + line + " already exists!")

'''
function: moves all files into 

Summary: creates direcories for each class in the label_file 

inputs: labels = label file created from the tensor training process
        output_dir = where you want to put the directories of the labels
returns: void

'''
def get_topk(result_file, n):
	file = open(result_file,"r")
	lines = file.readlines()
	topk = lines[1::n+1]
	files = lines[0::n+1]
	classes = []
	for i in range(len(topk)):
		classes.append(topk[i].split(None,1)[0])
	return files, classes
'''
function: moves all files into 

Summary: creates direcories for each class in the label_file 

inputs: labels = label file created from the tensor training process
        output_dir = where you want to put the directories of the labels
returns: void

'''
def movefiles(files,classes,output_dir):
	names = []
	for file in files:
		names.append(os.path.basename(file))
	for i in range(len(files)):
		file = str(files[i])
		print(file)
		shutil.move(file[:-1], os.path.join(str(output_dir) + "/" + str(classes[i]), str(names[i]))) 
'''
function: moves all files into 

Summary: creates direcories for each class in the label_file 

inputs: labels = label file created from the tensor training process
        output_dir = where you want to put the directories of the labels
returns: void

'''
def organize_files(label_file, output_dir, result_file):
	classes = load_labels(label_file)
	n = len(classes)
	label_classes = open(label_file, "r")
	files, classes  = get_topk(result_file,n)
	mkdir(label_classes, output_dir)
	movefiles(files, classes, output_dir)