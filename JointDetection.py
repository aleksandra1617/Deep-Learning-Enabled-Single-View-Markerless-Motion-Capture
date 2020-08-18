# Confidence and affinity maps are outputted from the first layer.
# https://www.geeksforgeeks.org/python-docstrings/
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense

from numpy import asarray
from numpy import unique
from numpy import argmax

import numpy as np
import cv2


class JointDetector:
    # Private Fields
    __instance = None
    __default_model = None

    @staticmethod
    def get_instance():
        if JointDetector.__instance is None:
            JointDetector()
        return JointDetector.__instance

    def __init__(self):
        # Public fields
        self.CONFIG = {"NumJoints": None, "FrameShape": None, "BatchSize": None, "Epochs": None}

        """ Virtually private constructor. """
        if JointDetector.__instance is not None:
            raise Exception("Instancing not permitted, the class KeypointDetector is a Singleton! " +
                            "Please call the get_instance function instead!")
        else:
            JointDetector.__instance = self

    # region CONFIG SECTION
    def configure(self, joint_data, frame_shape, batch_size, num_epochs, default_model=None):
        """
        Sets the configuration of the model at start.

        Parameters:
        ----------
        (object) default_model: a pre-trained or pre-configured model.

        Returns:
        -------
        Configured Convolutional Neural Network Model.
        """

        if default_model is not None:
            JointDetector.get_instance().model = default_model
        else:
            # set configuration
            JointDetector.get_instance().CONFIG["NumJoints"] = joint_data
            JointDetector.get_instance().CONFIG["FrameShape"] = frame_shape
            JointDetector.get_instance().CONFIG["BatchSize"] = batch_size
            JointDetector.get_instance().CONFIG["NumEpochs"] = num_epochs

            channel_dimension = 1  # TODO: Test with 3

            #convolution
            model = Sequential()
            model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=self.CONFIG["FrameShape"]))
            model.add(BatchNormalization(axis=channel_dimension))
            model.add(MaxPooling2D(pool_size=(8, 8)))
            # model.add(Dropout(0.25))
            #
            model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
            model.add(BatchNormalization(axis=channel_dimension))
            model.add(MaxPooling2D(pool_size=(4, 4)))
            # model.add(Dropout(0.25))

            model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
            # model.add(BatchNormalization(axis=channel_dimension))

            model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
            model.add(BatchNormalization(axis=channel_dimension))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            #model.add(Dropout(0.25))
            model.add(Flatten())
            # model.add(Dense(1024))
            # model.add(Activation("softmax"))
            model.add(Dense(128))
            model.add(Activation("relu"))
            #model.add(BatchNormalization())
            #model.add(Dropout(0.5))
            model.add(Dense(self.CONFIG["NumJoints"],))
            model.add(Activation("sigmoid"))
            model.summary()

            JointDetector.get_instance().model = model
    #endregion

    def create_model(self, train):
        model = JointDetector.get_instance().model

        # Define loss, optimizer, train and test the model
        model.compile(optimizer='adadelta', loss='mse', metrics=['mae', 'mse'])

        # TODO: Train on all the available videos.
        model.fit(train["input"][0], train["output"][0], epochs=self.CONFIG["NumEpochs"], batch_size=self.CONFIG["BatchSize"])
        #JointDetector.get_instance().predict(test["input"][0])
        #loss, acc = model.evaluate(test["input"][0], test["output"][0], verbose=0)
        #print('Accuracy: %.3f' % acc)

    def predict(self, input_data):
        predicted_output = []
        for image in input_data:
            img_array = asarray([image])

            # Make a prediction and save it in the list.
            predicted_output.append(JointDetector.get_instance().model.predict(img_array))

        return predicted_output

    def object_tracking(self):
        """
        Improves the accuracy of motion extraction as it trains a network per joint instead of per person.

        """
        pass

    def object_detection(self):
        """
        Used to add functionality to object tracking algorithms allowing them to track multiple objects at once.
        In this case that would be multiple people.

        """
        pass
