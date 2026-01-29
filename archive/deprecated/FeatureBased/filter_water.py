import numpy as np
from extract import CNNFeatureExtractor
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import h5py
import os
import joblib
import glob

out_path = os.path.abspath('output/')
le = joblib.load(os.path.join(out_path, 'le.joblib'))
sc = joblib.load(os.path.join(out_path, 'sc.joblib'))
data_file = h5py.File(os.path.join(out_path, 'labeled.h5'), 'r')
features = np.array(data_file['features'])
print(features.shape)
labels = np.array(data_file['labels'])
data_file.close()

water_dir = '/home/sam/Documents/e4e/mvnm_feature_based/dataset/water'
files = glob.glob(water_dir+'/*.jpg')
water_features = []
model = CNNFeatureExtractor()
for f in files:
    img = image.load_img(f)
    img = image.img_to_array(img)
    feat = model.extract(img)[0]
    # feat = np.random.randn(512)
    water_features.append(feat)

water_features = np.array(water_features)
len_old = features.shape[0]
features = np.vstack((features, water_features))
# print(len_old)
reduced = TSNE(n_components=2).fit_transform(features).T
print(reduced.shape)
m_reduced = reduced[:,0:len_old]
w_reduced = reduced[:,len_old:]
# print(m_reduced.shape)
plt.figure()
plt.subplot(211)
plt.scatter(m_reduced[0], m_reduced[1])
plt.subplot(212)
plt.scatter(w_reduced[0], w_reduced[1])
plt.show()
