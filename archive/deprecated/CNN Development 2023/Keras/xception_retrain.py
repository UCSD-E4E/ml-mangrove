import sys
import os
import argparse
import warnings
import tensorflow as tf
from keras.layers import *
from keras.optimizers import *
from keras.models import Model
from keras.models import load_model
from keras import backend as k
from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)



#Train model based on input imagery wiith training and test data

#def train(train_data_dir, output_model, epoch_num = 100, img_width=256):

def train(train_data_dir, epoch_num, img_width, test_split, output_model):
	img_width=256

	base_model = Xception(include_top=False, weights='imagenet', input_shape = (img_width,img_width,3))
	
	#Freeze all previously trained layers in the base model
	


	#Set up input layers and pretrained input layers
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
	predictions = Dense(5, activation='softmax')(x)

	for layer in base_model.layers[:]:
		layer.trainable = False

	model = Model(base_model.input, predictions)
	model.summary()

	transformation_ratio = 0.2
	batch_size = 32
	print("Creating train and test set generators")
	train_datagen = ImageDataGenerator(rescale=1. / 255,
										rotation_range=transformation_ratio,
										shear_range=transformation_ratio,
										zoom_range=transformation_ratio,
										cval=transformation_ratio,
										horizontal_flip=True,
										vertical_flip=True,
										validation_split = test_split)

	validation_datagen = ImageDataGenerator(rescale=1. / 255)

	print("Getting Training Data from Directory")
	train_generator = train_datagen.flow_from_directory(train_data_dir,
														target_size=(img_width, img_width),
														batch_size=batch_size,
														class_mode='categorical')
    
	print("Getting Validation Data from Directory")
	validation_generator = validation_datagen.flow_from_directory(train_data_dir,
																	target_size=(img_width, img_width),
																	batch_size=batch_size,
																	class_mode='categorical')

	print("Compiling model...")
	model.compile(loss="categorical_crossentropy", optimizer='nadam',metrics=["accuracy"])

	print("Training model...")
	history = model.fit_generator(train_generator, epochs=30,validation_data=validation_generator, steps_per_epoch=train_generator.samples/train_generator.batch_size, validation_steps = validation_generator.samples/validation_generator.batch_size) 
	
	print("Saving trained model")
	model.save(output_model)

	return history

if __name__ == "__main__":
	#take in input arguments from the command line
	parser = argparse.ArgumentParser(description="Retrain Keras.applications models on your own imagery")
	parser.add_argument("--path", help = "Path of training imagery you want to use")
	parser.add_argument("--output_model", help = "location at which you want the model file to be saved at")
	parser.add_argument("--train_split", help = "Percentage of training imagery in the train/test split e.g. 0.8 for an 80/20 train/test split", default=0.8)
	parser.add_argument("--epoch_num", help="Number of epochs you want to train your model for.",default=100)
	parser.add_argument("--image_width", help="width of your input tile imagery",default=256)
	parser.add_argument("--transformation_ratio", help="", default=0.2)
	args = parser.parse_args()

	if args.path:
		path = args.path
	else:
		print("No training imagery path was given, please check your --path argument")
		sys.exit()

	if args.output_model:
		output_model = args.output_model
		if os.path.splitext(output_model)[1] != '.h5':
			print("Output file name does not use .h5 extension, check your --output_model argument")
			sys.exit()
	else:
		print("WARNING: No output model has been set, check your --output_model argument ")

	if args.train_split:
		train_percent = round(float(args.train_split),2)
		if not  (float(0) <= train_percent) and (float(1) >= train_percent):
			print("Train split is not between 0 and 1, please check your --train_split argument")
			sys.exit()
		test_percent = round((1 - train_percent),2)

	if args.epoch_num:
		epoch_num = args.epoch_num

	if args.image_width:
		image_width = args.image_width

	



	history = train(path, epoch_num, image_width, test_percent, output_model)