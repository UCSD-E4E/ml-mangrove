import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import os, argparse, joblib
import shutil
from tqdm import tqdm
import yaml
import sys

# 95%m, 97%nm site 7 recall with sparse categorical cross-entropy and 10 epochs
# 76%m, 98%nm site 8 recall w/ 256 neurons, up to ~82%m recall w/ 0.1 dropout
# presence of water as a separate class improves m recall by 1-2% on average
# goes to 83%m recall on site 8 w/ 2 dense layer-dropout pairs
# Training on site 8 is >95% for 7 and 9, but bad for the training set even w/o water
# Training on sites 4 and 8, with water from site 4, gives consistent ~94% accuracy

# When I mix all of the data together and then train on half, we get ~98% for everthing when testing
# on the other half (with water). The variance of each metric is also an order of magnitude smaller.
# Water does not seem to affect the accuracy significantly, but I should test this rigourously.
# It doesn't make a difference if multiple scalings are present, which is a bit worrying.
# Also doesn't matter if unlabeled data is mixed in as mangrove. It was site 10, which is unique, so it 
# may have only affected the site 10 classifications.

# Dropout layers reduce accuracy by about 0.4%
# Removing the second dense layer is not statistically significant
# Overall: Training set is by far the most important variable

# 310 neurons, 2 dropouts at 0.3:
# train on site7-11 gets 88-95% accuracy on site 1, which is below the Inceptionv3 96%
# train on dataset/train gets up to 93% accuracy on output
# train with extracted Inceptionv3 gets about the same all around

def remove_water(x_test, y_test, le):
    '''
    Remove vectors labeled as water from a dataset.
    '''

def create_model():
    '''
    Create an MLP model with input of a given size.
    '''
    inputs = tf.keras.Input(shape=(2048,))
    x = layers.Dense(310, activation='relu')(inputs)
    x = layers.Dropout(rate=0.3, noise_shape=(310,))(x)
    x = layers.Dense(310, activation='relu')(x)
    x = layers.Dropout(rate=0.3, noise_shape=(310,))(x)
    outputs = layers.Dense(labels.shape[1], activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(momentum=0.1),
        metrics=['accuracy'])
    return model
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train directory', default='output-site8/')
    parser.add_argument('--test', help='test directory')
    parser.add_argument('-r', '--retrain', action='store_true', help='retrain the model')
    parser.add_argument('--indir', help='directory of unlabeled images')
    parser.add_argument('--outdir', help='directory to output sorted images')
    parser.add_argument('--sort', action='store_true', help='sort unlabeled images')
    parser.add_argument('-v', '--validate', action='store_true', help='test on preprocessed data')
    parser.add_argument('--xv', action='store_true', help='cross-validate')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--cfg', help='config file')
    args = parser.parse_args()
    print('Are the train and test sets normalized the same way?')

    train_path = os.path.abspath(args.train)
    test_path = os.path.abspath(args.test)
    features = np.load(os.path.join(train_path, 'features.npy'))
    sc_train = joblib.load(os.path.join(train_path, 'sc.joblib'))
    features = sc_train.inverse_transform(features)
    labels = np.load(os.path.join(train_path, 'labels.npy'))

    # Load test set
    le_train = joblib.load(os.path.join(train_path, 'le.joblib'))
    x_test = np.load(os.path.join(test_path, 'features.npy'))
    y_test = np.load(os.path.join(test_path, 'labels.npy'))

    # Load the training set. The data from each folder will be un-normalized with the scaler if it is
    # present, or assumed to be un-normalized otherwise.
    labels = []
    features = []
    for tp in train_paths:
        tp_labels = np.load(os.path.join(tp, 'labels.npy'))
        labels.append(tp_labels)
        if os.path.isfile(os.path.join(tp, 'sc.joblib')):
            sc_train = joblib.load(os.path.join(tp, 'sc.joblib'))
            features.append(sc_train.inverse_transform(np.load(os.path.join(tp, 'features.npy'))))
        else:
            features.append(np.load(os.path.join(tp, 'features.npy')))
    labels = np.concatenate(labels)
    features = np.vstack(features)

    if os.path.isfile(os.path.join(test_path, 'sc.joblib')):
        sc_test = joblib.load(os.path.join(test_path, 'sc.joblib'))
        x_test = sc_test.inverse_transform(x_test)
    
    # Randomize the order of the training set so that the validation set is different each time
    data = np.hstack([features, labels.reshape(-1, 1)])
    np.random.shuffle(data)
    features = data[:,:-1]
    labels = data[:,-1:]

    # Load test set
    le_train = joblib.load(os.path.join('output/', 'le.joblib'))
    x_test = np.load(os.path.join(test_path, 'features.npy'))
    y_test = np.load(os.path.join(test_path, 'labels.npy'))
    sc_test = joblib.load(os.path.join(test_path, 'sc.joblib'))
    x_test = sc_test.inverse_transform(x_test)

    y_test = np.minimum(y_test, np.ones_like(y_test))    # label water (2) as nm (1)
    labels = np.minimum(labels, np.ones_like(labels))

    # Binarize class labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels.reshape(-1, 1))
    if not args.sort:
        y_test = lb.transform(y_test.reshape(-1, 1))
        print(y_test)
    
    if args.retrain:
        # Retrain the model
        model = create_model(features.shape[1])
        history = model.fit(features, labels, batch_size=32, epochs=10)
    else:
        # Load saved model
        model = tf.keras.models.load_model(os.path.join(train_path, 'fc_model.h5'))
    print(model.summary())
    if args.validate:
        y_pred = np.round(model.predict(x_test))    # Round sigmoid output to 0 or 1
        cm = confusion_matrix(y_test, y_pred)
        if np.unique(labels).shape[0] < 3:
            print(classification_report(y_test, y_pred, digits=6))
            print(cm)
        else:
            # Instead of using classification report, we compute our own statistics, due to the fact that water
            # is a subclass of nm, so water-nm misclassifications don't matter.
            n_samples = y_pred.shape[0]
            m_recall = cm[0,0]/np.sum(cm[0])
            nm_recall = np.sum(cm[1:, 1:])/np.sum(cm[1]+cm[2])     # sum of nm classified as nm and nm classified as water
            m_precision = cm[0, 0]/np.sum(cm[:,0])
            nm_precision = np.sum(cm[1:, 1:])/np.sum(cm[:,1:])
            print(cm)
            print('m recall:', m_recall)
            print('m precision:', m_precision)
            print('nm recall:', nm_recall)
            print('nm precision', nm_precision)
    elif args.sort:
        in_dir = os.path.abspath(args.indir)
        if args.outdir is None:
            out_dir = in_dir    # default to same location as in_dir
        else:
            out_dir = os.path.abspath(args.outdir)
        fnames = joblib.load(os.path.join(test_path, 'fnames.joblib'))
        y_pred = np.rint(model.predict(x_test)).astype(int)
        y_labels = le_train.inverse_transform(y_pred)
        for l in le_train.classes_:
            # Make class directories
            os.makedirs(os.path.join(in_dir, l), exist_ok=True)
        # Move all images to the correct class directory
        for i in tqdm(range(len(fnames))):
            src = os.path.join(in_dir, fnames[i])
            dst = os.path.join(out_dir, y_labels[i], fnames[i])
            try:
                shutil.move(src, dst)
            except FileNotFoundError:
                # pass
                print('no such file '+src)
    elif args.xv:
        # 10-fold cross-validation
        kfold = KFold(n_splits=10)
        cvscores = []
        for train_index, test_index in kfold.split(features):
            model = create_model(features.shape[1])
            x_train, x_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            model.fit(x_train, y_train, batch_size=64, verbose=0, epochs=10)
            scores = model.evaluate(x_test, y_test, verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1]*100)
        print('%.3f%% +/- %.3f%%' % (np.mean(cvscores), np.std(cvscores)))
    elif args.analyze:
        # Do PCA on the features
        pca = PCA()
        pca.fit(features)
        print(np.cumsum(pca.explained_variance_ratio_))
    if args.retrain:
        # Save the retrained model
        model.save(os.path.join(train_path, 'fc_model.h5'))
