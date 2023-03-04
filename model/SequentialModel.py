import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class SequentialModel:

    def __init__(self):
        self.model = None

    def build(self, img_height, img_width, num_classes):
        self.model = Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def save(self, save_path):
        self.model.save_weights(save_path)
        print("Model saved.")

    def load(self, load_path):
        load_op = self.model.load_weights(load_path).expect_partial()
        print('Model loaded.')
        return load_op