from pytocl.driver import Driver
from pytocl.car import State, Command
from keras.layers import Dense, concatenate, Input
from keras.models import Sequential, Model
import tensorflow as tf
import numpy as np


class Modelo:

    def crear_modelo(self, raycasts, velocity):

        # Capa de entrada
        layer1 = Input(shape=(6,))
        # Capas ocultas
        layer2 = Dense(9, activation='sigmoid')(layer1)
        layer3 = Dense(6, activation='sigmoid')(layer2)
        # Capa de salida
        output = Dense(3, activation='sigmoid')(layer3)

        '''
        ##outputSteer = Dense(1, activation='sigmoid')(layer3)
        ##outputVelocity = Dense(2, activation='relu')(layer3)
        ##output = concatenate([outputSteer, outputVelocity])
        '''

        model = Model(inputs=layer1, outputs=output)

        model.compile(loss='mse', optimizer='adam', metrics='accuracy')

        # Preprocesamiento
        r = np.array(raycasts)
        if velocity > 0:
            v = np.array([velocity])
        else:
            v = np.zeros(1)

        x = np.concatenate((r, v), axis=0)
        x = x.round(decimals=5)

        print(x)

        # result = model(x)

        # tf.keras.utils.plot_model(model, to_file="tmp/model_shape_info2.png", show_shapes=True, show_layer_names=True)
