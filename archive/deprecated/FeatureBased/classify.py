import numpy as np
from extract import CNNFeatureExtractor
import h5py
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE, MDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
import argparse
import glob


# 78%m recall on site 8
# 89%m recall on site 7
# 67%m recall on site 9
# Overall, when trained on site 8 and my custom set (4 and water from Dillon), slightly less precise in 3-class
# Also takes an order of magnitude longer to train, and prior research suggests it is less accurate than an NN
# Good to note, but I think I will stick to the NN

def grid_search_params(grid, svc, data, target):
    clf = GridSearchCV(svc, grid, cv=5)
    print('[STATUS] Beginning grid search...')
    clf.fit(data, target)
    print(clf.best_params_)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xv', action='store_true', help='10-fold cross-validation')
    parser.add_argument('-v', '--validate', action='store_true', help='validate the model on a dataset')
    parser.add_argument('-m', '--model', default='svm', help = 'the model to use (svm, rf, knn')
    parser.add_argument('--train', help='train directory')
    parser.add_argument('--test', help='test directory')
    parser.add_argument('-r', '--retrain', action='store_true', help='retrain the model and save it')
    parser.add_argument('--show', action='store_true', help='show the images with their predicted labels')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--gs', action='store_true', help='grid search SVM params')
    parser.add_argument('--vp', action='store_true', help='validate preprocessed')
    args = parser.parse_args()
    if args.show:
        import cv2

    train_path = os.path.abspath(args.train)
    test_path = os.path.abspath(args.test)
    train_labels = os.listdir(train_path)
    extractor = CNNFeatureExtractor()
    if not args.xv:
        le = joblib.load(os.path.join(train_path, 'le.joblib'))
        sc = joblib.load(os.path.join(train_path, 'sc.joblib'))
    features = np.load(os.path.join(train_path, 'features.npy'))
    features2 = np.load(os.path.abspath('output-site8/features.npy'))
    labels = np.load(os.path.join(train_path, 'labels.npy'))
    labels2 = np.load(os.path.abspath('output-site8/labels.npy'))
    features = np.vstack([features, features2])
    labels = np.concatenate([labels, labels2])

    if args.retrain:
        if args.model=='rf':
            clf = RandomForestClassifier(n_estimators=300, random_state=4)
        elif args.model=='knn':
            clf = KNeighborsClassifier()
        elif args.model=='svm':
            clf = SVC(gamma='auto')
        print(labels.shape)
        print(features.shape)
        clf.fit(features, labels)
    else:
        if args.model=='rf':
            clf = joblib.load(os.path.join(train_path, 'rf.joblib'))
        elif args.model=='knn':
            clf = joblib.load(os.path.join(train_path, 'knn.joblib'))
        elif args.model=='svm':
            clf = joblib.load(os.path.join(train_path, 'svm.joblib'))

    if args.xv:
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(clf, features, labels, cv=kfold, scoring='accuracy')
        msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
        print(msg)
    elif args.validate:
        in_dirs = os.listdir(test_path)
        in_dirs.sort()
        for d in in_dirs:
            images = glob.glob(os.path.join(test_path, d, '*.jpg',))
            im_count = 0
            correct = 0
            for f in images:
                img = image.load_img(f)
                img = image.img_to_array(img)
                img_bgr = img[:,:,::-1].astype(np.uint8)
                feature = extractor.extract(img)
                feature = sc.transform(feature.reshape(1, -1))
                prediction = clf.predict(feature)
                prediction_label = le.inverse_transform(prediction)[0]
                im_count += 1
                if d == prediction_label or (d=='nm' and prediction_label=='water'):
                    correct += 1
                if im_count % 10 ==0:
                    print(im_count)
                if args.show:
                    # print(clf.predict_proba(feature))
                    cv2.putText(img_bgr, prediction_label, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
                    cv2.imshow('image', img_bgr)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
            print('Actual {}: {}/{} ({}%) labeled as {}'.format(d, correct, im_count, correct/im_count*100, d))
    elif args.analyze:
        import matplotlib.pyplot as plt
        # pca = PCA()
        # pca.fit(features)
        # print(np.cumsum(pca.explained_variance_ratio_))
        # for v in features:
        #     if np.allclose(v, np.zeros_like(v)):
        #         print(v)
        labels = le.inverse_transform(labels)
        reduced = TSNE(n_components=2, random_state=6).fit_transform(features)
        for l in list(le.classes_):
            print(l)
            m_reduced = reduced[labels==l].T
            plt.scatter(m_reduced[0], m_reduced[1])
        plt.show()
        # print(features)
    elif args.gs and args.model=='svm':
        grid = {'gamma':[0.0001, 0.001, 0.01, 0.1, 1], 'C':[0.1, 1, 10]}
        grid_search_params(grid, svc=clf, data=features, target=labels)
    elif args.vp:
        x_test = np.load(os.path.join(test_path, 'features.npy'))
        y_test = np.load(os.path.join(test_path, 'labels.npy'))
        y_pred = clf.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
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
        print(cm)
                
    if args.retrain:
        if args.model=='rf':
            joblib.dump(clf, os.path.join(train_path, 'rf.joblib'))
        elif args.model=='knn':
            joblib.dump(clf, os.path.join(train_path, 'knn.joblib'))
        elif args.model=='svm':
            joblib.dump(clf, os.path.join(train_path, 'svm.joblib'))
    