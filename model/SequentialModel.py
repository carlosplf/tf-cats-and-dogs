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
        model = VGG16(include_top=False, input_shape=(img_height, img_width, 3))
        
        # Mark loaded layers as not trainable
        for layer in model.layers:
            layer.trainable = False
        
        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
        output = Dense(1, activation='sigmoid')(class1)
        
        model = Model(inputs=model.inputs, outputs=output)

        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        
        self.model = model

    def build(self, img_height, img_width, num_classes):
        model = Sequential([
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
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