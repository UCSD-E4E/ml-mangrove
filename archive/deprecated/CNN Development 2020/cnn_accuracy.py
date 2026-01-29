import os 
import argparse 

def get_topk(result_file):
	lines = result_file.readlines()
	topk = lines[1::4]
	classes = []
	for i in range(len(topk)):
		classes.append(topk[i].split(None,1)[0])
	return classes

def test_accuracy(classes, class_name):
	num_correct = 0 
	for i in range(len(classes)):
		if classes[i] == class_name:
			num_correct = num_correct + 1
	accuracy = float(num_correct) / float(len(classes))
	print(str(num_correct) + "/" + str(len(classes)))
	return accuracy

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--class_name", help="name of class to test accuracy on")
	parser.add_argument("--results", help="name of results file")
	args = parser.parse_args()
	if args.class_name:
		classoi = args.class_name
	if args.results:
		file = args.results
	result_file =  open(file,"r")
	class_list = get_topk(result_file)
	print(test_accuracy(class_list, classoi))


