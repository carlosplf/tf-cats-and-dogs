import tensorflow as tf
import os
import argparse
import logging
import threading
import json

from random import randrange
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
STATUS_FILES_FOLDER = "status_control"
AVAILABLE_MODELS = ["vgg16", "custom"]


class APIMLRunner:
    def __init__(self):
        self.__config_log()

    def __config_log(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("debug.log"),
                logging.StreamHandler()
            ]
        )

    def create_train_dataset(self):
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

    def create_validation_dataset(self):
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

    def create_model(self, model_name):
        seq_model = SequentialModel()

        if model_name == "vgg16":
            logging.warning("VGG16 model uses a significant bigger amount of memory. Check hardware and batch size.")
            seq_model.build_vgg16(IMG_HEIGHT, IMG_WIDTH)
            return seq_model

        if model_name == "custom":
            seq_model.build(IMG_HEIGHT, IMG_WIDTH)
            return seq_model

        return None

    def train_model(self, n_epochs, seq_model, train_ds, val_ds):
        history = seq_model.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=n_epochs,
        )
        return history

    @staticmethod
    def _create_pid():
        max_pid_value = 1000000
        return randrange(max_pid_value)

    def run_training(self, model_name, n_epochs):

        logging.info("Starting training...")
       
        train_ds = self.create_train_dataset()
        val_ds = self.create_validation_dataset()

        seq_model = self.create_model(model_name)

        if seq_model is None:
            return None

        thread_pid = self._create_pid()

        new_thread = threading.Thread(
            target = self.training_function,
            args = (
                thread_pid,
                seq_model,
                n_epochs,
                train_ds,
                val_ds
            ),
            daemon=True
        )
        
        try:
            logging.debug("Starting a new thread for training...")
            new_thread.start()
            return thread_pid 
        except Exception as e:
            logging.error(str(e))
            return None

    @staticmethod
    def _write_status_to_file(pid, status):
        filename = STATUS_FILES_FOLDER + "/" + str(pid) + "_status.json"
        with open(filename, 'w') as f:
            json.dump(status, f)

    @staticmethod
    def get_pid_status(pid):
        filename = STATUS_FILES_FOLDER + "/" + str(pid) + "_status.json"
        try:
            with open(filename) as f:
                return json.load(f)
        except Exception as e:
            logging.warning("Error while searching for status pid " + str(pid))
            return None

    def training_function(self, thread_pid, seq_model, n_epochs, train_ds, val_ds):

        logging.info("Training function running...")
        logging.info("PID: " + str(thread_pid))

        status = {
            "status": "Running",
            "Message": "Training",
            "Pid": thread_pid
        }

        self._write_status_to_file(thread_pid, status)
        
        history = self.train_model(n_epochs, seq_model, train_ds, val_ds)
        
        seq_model.save(MODEL_SAVE_PATH) 

        csv_log_writer.write_log(history.history, CSV_LOG_FILE)

        logging.info("Finished training.")
        
        status = {
            "status": "Stopped",
            "Message": "Finished training",
            "Values": history.history,
            "Pid": thread_pid
        }

        self._write_status_to_file(thread_pid, status)
        return True

    def predict_from_file(self, seq_model, img_filename):
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
        
        del img_array
        del img
        del predictions
        del seq_model

        return score

    def run_predict(self, model_name, filename):

        logging.info("Predicting image...")

        seq_model = self.create_model(model_name)

        # Load model weights from Tensorflow saving.
        seq_model.load(MODEL_SAVE_PATH)
        
        if seq_model is None:
            return None
        
        return self.predict_from_file(seq_model, filename)

    def run_predict_all(self, model_name, folder_path):

        logging.info("Predicting all images...")

        seq_model = self.create_model(model_name)
        
        if seq_model is None:
            return None

        # Load model weights from Tensorflow saving.
        seq_model.load(MODEL_SAVE_PATH)
        
        for f in os.listdir(folder_path):
            self.predict_from_file(seq_model, folder_path + "/" + f)

