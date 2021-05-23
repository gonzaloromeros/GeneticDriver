from keras.layers import Dense, concatenate, Input
from keras.models import Sequential, Model
import tensorflow as tf
import numpy as np


class Modelo:

    def __init__(self):
        # Capa de entrada
        layer1 = Input(shape=(6,))

        # Capas ocultas
        layer2 = Dense(9, activation='sigmoid')(layer1)
        layer3 = Dense(6, activation='sigmoid')(layer2)

        # Capa de salida
        outputsteer = Dense(1, activation='tanh')(layer3)
        outputspeed = Dense(2, activation='sigmoid')(layer3)
        output = concatenate([outputsteer, outputspeed])

        # Crear arquitectura del modelo
        self.modelo = Model(inputs=layer1, outputs=output)

    def crear_modelo(self):
        ...

    def inferir_modelo(self, raycasts, speed):

        """
        # Capa de entrada
        layer1 = Input(shape=(6,))
        # Capas ocultas
        layer2 = Dense(9, activation='sigmoid')(layer1)
        layer3 = Dense(6, activation='sigmoid')(layer2)
        # Capa de salida
        output = Dense(3, activation='sigmoid')(layer3)


        ##output = Dense(3, activation='sigmoid')(layer3)


        ##model = Model(inputs=layer1, outputs=output)
        """

        # self.modelo.compile(loss='mse', optimizer='adam', metrics='accuracy')

        # Preprocesamiento
        r = np.array(raycasts)
        if speed > 0:
            v = np.array([speed])
        else:
            v = np.zeros(1)

        x = np.concatenate((r, v), axis=0)
        x = x.round(decimals=5)

        result = self.modelo.predict(x.reshape(1, 6), verbose=0)

        # tf.keras.utils.plot_model(self.modelo, to_file="tmp/model_shape_info.png", show_dtype=True, show_shapes=True, show_layer_names=True)

        return result
