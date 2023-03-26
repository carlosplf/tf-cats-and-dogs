import tensorflow as tf
import os
import argparse
import logging

from model.SequentialModel import SequentialModel
from csv_log_writer import csv_log_writer
from file_checker import file_checker


# Adjust TF log level and avoid INFO messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

DATA_DIR = "./PetImages"
BATCH_SIZE = 64
IMG_HEIGHT = 224
IMG_WIDTH = 224
MODEL_SAVE_PATH = "./model_save/weights"
CSV_LOG_FILE = "./logs/output_log.csv"


# Args for command line.
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", type=int,
                    help="Train the model using N epochs.")
parser.add_argument("--nosave",
                    help="Set no_save flag. Trained models won't be saved.",
                    action="store_true")
parser.add_argument("--vgg16",
                    help="Use the Keras VGG16 model.",
                    action="store_true")
parser.add_argument("-p", "--predict", type=str,
                    help="Predict an image class. -p <IMG_PATH>")
parser.add_argument("-pa", "--predict_all", type=str,
                    help="Predict all images inside a folder. -pa <FODLER_PATH>")
parser.add_argument("--check_images", type=str,
                    help="Check if images in specified folder are not corrupted.")
parser.add_argument("--debug",
                    help="Change log level to DEBUG.",
                    action="store_true")
args = parser.parse_args()



def create_train_dataset():
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=False,
        validation_split=0.25
    )
    train_ds = img_gen.flow_from_directory(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode='binary',
        subset="training",
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    return train_ds


def create_validation_dataset():
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=False,
        validation_split=0.25
    )

    val_ds = img_gen.flow_from_directory(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode='binary',
        subset="validation",
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    return val_ds


def create_model():
    seq_model = SequentialModel()

    if args.vgg16:
        logging.warning("VGG16 model uses a significant bigger amount of memory. Check hardware and batch size.")
        seq_model.build_vgg16(IMG_HEIGHT, IMG_WIDTH)
    else:
        seq_model.build(IMG_HEIGHT, IMG_WIDTH)
    return seq_model


def train_model(n_epochs, seq_model, train_ds, val_ds):
    history = seq_model.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=n_epochs
    )
    return history


def run_training(n_epochs):
    
    logging.info("Starting training...")
   
    train_ds = create_train_dataset()
    val_ds = create_validation_dataset()

    seq_model = create_model()

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
    Return:
        (float) classification score.
    """

    logging.info("Filename: " + img_filename)

    img = tf.keras.preprocessing.image.load_img(
        img_filename, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = seq_model.model.predict(img_array)

    score = predictions[0][0]*100

    if score > 50:
        logging.info("It's a Dog!. Probability: " + str(score) + "%")
    else:
        logging.info("It's a Cat!. Probability: " + str(score) + "%")
    
    return score


def run_predict(filename):

    logging.info("Predicting all images...")

    seq_model = create_model()

    # Load model weights from Tensorflow saving.
    seq_model.load(MODEL_SAVE_PATH)
    
    predict_from_file(seq_model, filename)


def run_predict_all(folder_path):

    logging.info("Predicting all images...")

    seq_model = create_model()

    # Load model weights from Tensorflow saving.
    seq_model.load(MODEL_SAVE_PATH)
    
    for f in os.listdir(folder_path):
        predict_from_file(seq_model, folder_path + "/" + f)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    
    if args.check_images:
        file_checker.check_images(args.check_images)
    
    if args.train:
        run_training(args.train)

    if args.predict:
        run_predict(args.predict)
    
    if args.predict_all:
        run_predict_all(args.predict_all)
