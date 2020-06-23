import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
import os
import argparse
from tqdm import tqdm

def load_data(path):
    '''
    Load labeled image data into arrays.

    Arguments:
        path: path to the directory
    
    Returns:
        np.ndarray: array of preprocessed images
        list: corresponding string labels
    '''
    labels = os.listdir(path)
    n_images = 0
    imshape = (256, 256, 3)

    for d in labels:
        n_images += len(os.listdir(os.path.join(path, d)))    # tally up all image files
    
    x = np.zeros((n_images, *imshape))    # input images
    y_str = []    # input labels as strings
    i = 0
    for d in labels:
        files = os.listdir(os.path.join(path, d))
        for f in tqdm(files):
            imfile = os.path.join(train_path, d, f)
            img = image.load_img(imfile)
            img = image.img_to_array(img)
            img = preprocess_input(img)
            x[i] = img
            y_str.append(d)
            i += 1
    return x, y_str


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train image directory')
    parser.add_argument('--test', help='test image directory')
    parser.add_argument('--save', help='save directory')
    parser.add_argument('--name', help='name to save or load model as (e.g. vgg16_1024)')
    parser.add_argument('-r', '--retrain', action='store_true', help='retrain the model')
    parser.add_argument('-l', '--labeled', action='store_true', help='use labeled data')
    parser.add_argument('--layer', help='layer to use')
    args = parser.parse_args()

    save_path = os.path.abspath(args.save)
    
    if args.retrain:
        # Build a new model, train it, and save it
        train_path = os.path.abspath(args.train)
        print(args.layer)
        full_model = VGG16(weights='imagenet', input_shape=(256, 256, 3), include_top=False)
        if args.layer is not None:
            base_model = Model(inputs=full_model.input, outputs=full_model.get_layer(args.layer).output)
        else:
            base_model = full_model
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dense(1024, activation='relu')(x)
        pred = layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=pred)
        try:
            model = tf.keras.utils.multi_gpu_model(model, gpus=2)
        except:
            print('[STATUS] Multiple GPUs not available, using CPU or single GPU...')
            model = tf.keras.models.Model(inputs=base_model.input, outputs=pred)
        
        for layer in base_model.layers:
            layer.trainable = False
        
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(), metrics=['acc'])

        x_train, y_train_str = load_data(train_path)
        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train_str)
        history = model.fit(x=x_train, y=y_train, batch_size=64, epochs=10)

        print('[STATUS] Saving model...')
        model.save(os.path.join(save_path, args.name+'.h5'))
        print('[STATUS] Saving label binarizer...')
        joblib.dump(lb, os.path.join(save_path, 'lb.joblib'))
    else:
        model = tf.keras.models.load_model(os.path.join(save_path, args.name+'.h5'))
        lb = joblib.load(os.path.join(save_path, 'lb.joblib'))
    
    if args.labeled:
        test_path = os.path.abspath(args.test)
        x_test, y_test_str = load_data(test_path)
        y_test = lb.transform(y_test_str)

        y_pred = np.argmax(model.predict(x_test), axis=1)
        cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        print(classification_report(np.argmax(y_test, axis=1), y_pred, digits=6))
        print(cm)
