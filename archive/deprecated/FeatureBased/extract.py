import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import numpy as np
import os
import argparse

class CNNFeatureExtractor:
    '''
    Class to represent a CNN-based feature extractor. This abstracts all the details of the CNN away from the classifier.
    '''
    def __init__(self, shape, layer=None):
        '''
        Initializes a CNNFeatureExtractor using the VGG16 CNN.

        Arguments:
            shape: the shape of the imput images
        '''
        if layer is None:
            self.model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=shape)
            # self.model = tf.keras.utils.multi_gpu_model(self.model, gpus=2)
            print(self.model.summary())
        else:
            full_model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=shape)
            self.model = Model(inputs=full_model.input, outputs=full_model.get_layer(layer).output)
    
    def extract(self, img):
        '''
        Uses the CNN to extract a feature vector from an image.

        Arguments:
            img: an image in array form. Must be the same shape as specified in the constructor.
        
        Returns:
            np.array: the feature vector, with shape (1, n), where n is the length of the CNN output
        '''
        img = preprocess_input(img)
        if len(img.shape)==3:
            img = np.expand_dims(img, axis=0)
        feature = self.model.predict(img)
        return feature

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batchsize', type=int, default=30, help='images per batch')
    parser.add_argument('--layer', help='the layer name to use (default last)')
    parser.add_argument('-i', '--input', help='path to input directory')
    parser.add_argument('-o', '--output', help='path to output directory')
    parser.add_argument('-f', '--savefnames', action='store_true', help='save filenames')
    parser.add_argument('-u', '--unlabeled', action='store_true', help='flag data as unlabeled')
    parser.add_argument('-s', '--side', type=int, help='tile side length, in px', default=256)
    args = parser.parse_args()
    imshape = (args.side, args.side, 3)

    extractor = CNNFeatureExtractor(shape=imshape)
    train_path = os.path.abspath(args.input)
    out_path = os.path.abspath(args.output)
    if not args.unlabeled:
        train_labels = os.listdir(train_path)
    else:
        train_labels = ['']     # if unlabeled, all images are in train_path directly
    batchsize = args.batchsize
    images_per_dir = max(map())

    labels = []
    features = []
    dirnum = 0      # directory number, 0 indexed
    fnames = []
    for d in train_labels:
        files = os.listdir(os.path.join(train_path, d))
        j = 0       # number of image in batch, 1 indexed
        i = 0       # number of image in directory, 1 indexed
        batch = []
        for f in files:
            i += 1
            imfile = os.path.join(train_path, d, f)
            img = image.load_img(imfile)
            img = image.img_to_array(img)
            if img.shape == imshape:
                # Only process the image if it is the right shape, to avoid some issues with tf.keras
                batch.append(img)
                labels.append(d)
                j += 1      # update j here so all batches but the last have batchsize images (for speed)
                if args.savefnames:
                    fnames.append(os.path.relpath(os.path.join(d, f)))
            else:
                print("{} is not {}, but is {}".format(imfile, imshape, img.shape))
            if j == batchsize or i == len(files):
                batch = np.array(batch)
                print(i)
                # dir_features[i-batch.shape[0]:i] = extractor.extract(batch)
                features.append(extractor.extract(batch))
                batch = []
                j = 0
            if i == images_per_dir:
                break
        dirnum += 1
    
    features = np.vstack(features)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    print('[STATUS] Saving data...')
    np.save(os.path.join(out_path, 'features.npy'), features)
    np.save(os.path.join(out_path, 'labels.npy'), labels)
    print('[STATUS] Saving label encoder...')
    joblib.dump(le, os.path.join(out_path, 'le.joblib'))
    if args.savefnames:
        joblib.dump(fnames, os.path.join(out_path, 'fnames.joblib'))