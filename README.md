# Tensorflow - Cats and Dogs

Tensorflow implementation to classify Cats and Dogs.

Dataset used can be found [HERE](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs).

## Installation

Make sure you have Python 3.8 and pip installed. The project contains two requirements files, being one for ROCM and other for Apple M1 silicon.

```
python3 -m venv ./env
source ./env/bin/activate
pip install -r <requirements-file.txt>
```

## How to run

```
usage: run.py [-h] [-t TRAIN] [--nosave] [--vgg16] [-p PREDICT] [-pa PREDICT_ALL] [--check_images CHECK_IMAGES] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN, --train TRAIN
                        Train the model using N epochs.
  --nosave              Set no_save flag. Trained models won't be saved.
  --vgg16               Use the Keras VGG16 model.
  -p PREDICT, --predict PREDICT
                        Predict an image class. -p <IMG_PATH>
  -pa PREDICT_ALL, --predict_all PREDICT_ALL
                        Predict all images inside a folder. -pa <FODLER_PATH>
  --check_images CHECK_IMAGES
                        Check if images in specified folder are not corrupted.
  --debug               Change log level to DEBUG.

```

Just run `run.py -t <n_epochs>` to train the model running all the images on the dataset, where <n_epochs> is the the number of times the software will go through the dataset.

You can use `run.py -p <img_path>` to predict a single image, trying to tell if it is a cat or a dog. If you have a folder with some photos of cats and dogs, you can use `run.py -pa <folder_path>` and the software will try to predict all the images inside this folder.

## Models

The software contains two CNN Sequential Model implementations. One is a custom Model build, based on the Keras VGG16 implementation, and the
other is the [original Keras VGG16 Model](https://keras.io/api/applications/vgg/).

To use the original Keras VGG16 Model, just use the `--vgg16` argument when calling `run.py`.
