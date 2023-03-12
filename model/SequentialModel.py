import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten


class SequentialModel:

    def __init__(self):
        self.model = None

    def build_vgg16(self, img_height, img_width):
        """
        Build a Model using the VGG16 Model from Keras, with the 'imagenet' weights.
        Args:
            img_height, img_width: image target size.
        Return: None.
        """
        print("Building VGG16 model...")
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
        base_model.trainable = False
        
        model = Sequential([
            base_model,
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        opt = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        
        self.model = model

    def build(self, img_height, img_width):
        print("Building custom model...")
        model = Sequential([
            layers.Conv2D(input_shape=(img_height,img_width,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.model = model

    def save(self, save_path):
        self.model.save_weights(save_path)
        print("Model saved.")

    def load(self, load_path):
        load_op = self.model.load_weights(load_path).expect_partial()
        print('Model loaded.')
        return load_op
