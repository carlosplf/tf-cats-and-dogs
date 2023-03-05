import tensorflow as tf
import os
import argparse
import numpy as np

from model.SequentialModel import SequentialModel
from csv_log_writer import csv_log_writer


# Adjust TF log level and avoid INFO messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

DATA_DIR = "./PetImages"
BATCH_SIZE = 64
IMG_HEIGHT = 120
IMG_WIDTH = 120
MODEL_SAVE_PATH = "./model_save/weights"
CSV_LOG_FILE = "./logs/output_log.csv"


# Args for command line.
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", type=int,
                    help="Train the model using N epochs.")
parser.add_argument("--nosave",
                    help="Set no_save flag. Trained models won't be saved.",
                    action="store_true")
parser.add_argument("-p", "--predict", type=str,
                    help="Predict an image class. -p <IMG_PATH>")
parser.add_argument("-pa", "--predict_all", type=str,
                    help="Predict all images inside a folder. -pa <FODLER_PATH>")
args = parser.parse_args()



def create_train_dataset():
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=False,
        validation_split=0.2
    )
    train_ds = img_gen.flow_from_directory(DATA_DIR, batch_size=BATCH_SIZE, shuffle=True, class_mode='binary', subset="training", target_size=(IMG_HEIGHT, IMG_WIDTH))
    return train_ds


def create_validation_dataset():
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=False,
        validation_split=0.2
    )

    val_ds = img_gen.flow_from_directory(DATA_DIR, batch_size=BATCH_SIZE, shuffle=True, class_mode='binary', subset="validation", target_size=(IMG_HEIGHT, IMG_WIDTH))
    return val_ds


def create_model(num_classes):
    seq_model = SequentialModel()
    seq_model.build_vgg16(IMG_HEIGHT, IMG_WIDTH)
    return seq_model


def train_model(n_epochs, seq_model, train_ds, val_ds):
    history = seq_model.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=n_epochs
    )
    return history


def run_training(n_epochs):
    
    print("Starting training...")
   
    train_ds = create_train_dataset()
    val_ds = create_validation_dataset()
    num_classes = len(list(train_ds.class_indices.keys()))

    seq_model = create_model(num_classes)

    history = train_model(n_epochs, seq_model, train_ds, val_ds)
    
    if not args.nosave:
        seq_model.save(MODEL_SAVE_PATH) 

    csv_log_writer.write_log(history.history, CSV_LOG_FILE)

    return history.history


def predict_from_file(seq_model, img_filename):
    """
    Load an image and predict using the trained Model.
    Args:
        seq_model: SequentialModel class instance.
        img_filename: name of the img file to be loaded.
    Return: Most likely class and the classification score.
    """

    print("Filename: ", img_filename)

    img = tf.keras.preprocessing.image.load_img(
        img_filename, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = seq_model.model.predict(img_array)

    score = predictions[0][0]*100

    if score > 0.5:
        print("It's a Dog!. Probability: ", score, "%")
    else:
        print("It's a Cat!. Probability: ", score, "%")


def run_predict(filename):

    print("Predicting all images...")

    num_classes = 2
    seq_model = create_model(num_classes)

    # Load model weights from Tensorflow saving.
    seq_model.load(MODEL_SAVE_PATH)
    
    predict_from_file(seq_model, filename)


def run_predict_all(folder_path):

    print("Predicting all images...")

    num_classes = 2
    seq_model = create_model(num_classes)

    # Load model weights from Tensorflow saving.
    seq_model.load(MODEL_SAVE_PATH)
    
    for f in os.listdir(folder_path):
        predict_from_file(seq_model, folder_path + "/" + f)


if __name__ == "__main__":
    
    if args.train:
        run_training(args.train)

    if args.predict:
        run_predict(args.predict)
    
    if args.predict_all:
        run_predict_all(args.predict_all)