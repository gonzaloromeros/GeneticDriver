from pytocl.driver import Driver
from pytocl.car import State, Command
from keras.layers import Dense, concatenate, Input
from keras.models import Sequential, Model
import tensorflow as tf
import numpy as np



class Modelo():

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

        tf.keras.utils.plot_model(model, to_file="tmp/model_shape_info.png", show_shapes=True, show_layer_names=True)