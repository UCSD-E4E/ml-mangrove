from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
import os
import time

# for performance analysis
start_time = time.time()

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
		# locks graph so no new operations can be added
		graph.finalize()
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

def main():
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
	parser.add_argument("--output_file", help="path and name with which to write result file")
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
	if args.output_file:
		output_file = args.output_file
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
	result_file = open(output_file,"w")
	file_list = []
	for root, dirs, files in os.walk(os.path.abspath(image_directory)):
   		for file in files:
                        file_list.append(os.path.join(root, file))

	input_name = "import/" + input_layer
	output_name = "import/" + output_layer
	input_operation = graph.get_operation_by_name(input_name)
	output_operation = graph.get_operation_by_name(output_name)

	count = 0
	batch_time = time.time()
	batch_times = []
	labels = load_labels(label_file)
	labels.append('file')
	result_csv = pd.DataFrame(columns=labels)
	#process imagery
	for file in file_list:
		if file.endswith(".jpg"):
			t = read_tensor_from_image_file(
				file,
				input_height=input_height,
				input_width=input_width,
				input_mean=input_mean,
				input_std=input_std)
			cur_result = pd.DataFrame(columns=labels)
			with tf.Session(graph=graph) as sess:
				results = sess.run(output_operation.outputs[0], {
				input_operation.outputs[0]: t})
			results = np.squeeze(results)

			top_k = results.argsort()[-5:][::-1]
			cur_result.set_value(0, 'file', file)  
			for i in top_k:
				cur_result.set_value(0, labels[i],  result[i])
			print(cur_result)
			result_csv = result_csv.append(cur_result)
			result_csv.to_csv('results.csv')
			
			# for logs
			count += 1
			if(count%100 == 0): 
				print("\n\nNum images processed: {}".format(count))
				batch_times.append(time.time()-batch_time)
				print("Time for last 100 images: {}\n\n".format(time.time()-batch_time))
				batch_time = time.time()
	
	# final logs
	print("Script took %s seconds to execute" % (time.time() - start_time))
	print("Batch times: ")
	print(batch_times)
	result_file.close()

if __name__ == "__main__":
	main()