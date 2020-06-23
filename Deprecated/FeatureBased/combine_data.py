import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import joblib

outputs = glob.glob('output*')
features = []
labels = []

for d in outputs:
    if os.path.isdir(d):
        sc = joblib.load(os.path.join(d, 'sc.joblib'))
        le = joblib.load(os.path.join(d, 'le.joblib'))
        print(d, le.classes_)
        if(len(le.classes_) > 1):
            features.append(sc.inverse_transform(np.load(os.path.join(d, 'features.npy'))))
            labels.append(np.load(os.path.join(d, 'labels.npy')))

features = np.concatenate(features, axis=0)
labels = np.hstack(labels)
print(features.shape, labels.shape)
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.5)
np.save('dataset/train/features.npy', x_train)
np.save('dataset/test/features.npy', x_test)
np.save('dataset/train/labels.npy', y_train)
np.save('dataset/test/labels.npy', y_test)