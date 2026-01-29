# cnn-features
## Introduction
This is a feature-based classifier for drone photos of mangroves swamps. `extract.py` uses the pre-trained VGG16 CNN to convert image tiles into a representative feature vector. These feature vectors can then be used to train other machine learning algorithms, such as support vector machines or other neural networks.

## Dependencies
- `tensorflow`
- `keras`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tqdm`
- `cv2` (if using `classify.py`, which you shouldn't be)

## How to use
### Docker
A Dockerfile is provided to build a Docker container for running these scripts with GPU Tensorflow. `extract.py` should be run this way, especially if using Microsoft Azure as we are. From this repo's top-level directory, the command is:
```
$ docker build -t features .
```

### `extract.py`
This is the first step to any classification. `extract.py` converts a directory full of image tiles into a list of feature vectors. The recommended way to do this is with `runner.sh`, which will use the Docker container on GPU. The command is:
```
$ ./runner.sh <path to output directory with respect to pwd> <absolute path to input directory>
```
The reason one path is relative and another is absolute is to allow for effective use of tab completes, as input directory names can be quite long.

Arguments:
- `-n <number of batches to use per directory>`
- `-s <tile side length, in px >`

#### `runner.sh`
The only time you should need to modify this file is if you need to change the input tile size or switch between labeled and unlableled data.

### `train_fc.py`
This script uses the extracted features to classify tiles.