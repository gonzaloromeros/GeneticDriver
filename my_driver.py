from pytocl.driver import Driver
from pytocl.car import State, Command

from keras.layers import Dense, concatenate, Input
from keras.models import Sequential, Model
import tensorflow as tf

from model import Modelo

import numpy as np


class MyDriver(Driver):

    def drive(self, carstate: State) -> Command:
        command = Command()

        command.accelerator = 1
        command.gear = 1
        command.brake = 0

        raycasts = np.array(carstate.focused_distances_from_edge)

        if np.average(raycasts) != -1:
            print(raycasts)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        Modelo.create_model(self)

        return command

'''
    def create_model(self):

        #Capa de entrada
        layer1 = Input(shape=(6,))
        #Capas ocultas
        layer2 = Dense(9, activation='sigmoid')(layer1)
        layer3 = Dense(6, activation='sigmoid')(layer2)
        #Capa de salida
        outputSteer = Dense(1, activation = 'sigmoid')(layer3)
        outputVelocity = Dense(2, activation = 'relu')(layer3)
        output = concatenate([outputSteer, outputVelocity])

        model = Model(inputs = layer1, outputs = output)

        model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

        #tf.keras.utils.plot_model(model, to_file="model_with_shape_info.png", show_shapes=True, show_layer_names=True)
'''