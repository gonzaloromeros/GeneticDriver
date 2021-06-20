from keras.layers import Dense, concatenate, Input
from keras.models import Sequential, Model
import tensorflow as tf
import numpy as np
import os


class Modelo:

    def __init__(self, generation, n):
        # Capa de entrada
        layer1 = Input(shape=(6,))

        # Capas ocultas
        layer2 = Dense(9, activation='sigmoid', use_bias=True, bias_initializer='RandomNormal')(layer1)
        layer3 = Dense(6, activation='sigmoid', use_bias=True, bias_initializer='RandomNormal')(layer2)
        layer4 = Dense(4, activation='sigmoid', use_bias=True, bias_initializer='RandomNormal')(layer3)

        # Capa de salida
        outputsteer = Dense(1, activation='tanh', use_bias=True, bias_initializer='RandomNormal')(layer4)
        #outputspeed = Dense(1, activation='sigmoid', use_bias=True, bias_initializer='RandomNormal')(layer3)
        outputspeed = Dense(1, activation='tanh', use_bias=True, bias_initializer='RandomNormal')(layer4)
        output = concatenate([outputsteer, outputspeed])

        # Crear arquitectura del modelo
        self.modelo = Model(inputs=layer1, outputs=output)

        if generation == 1:
            self.modelo.set_weights(self.inicializar_pesos())
        elif generation != -1:
            self.modelo.set_weights(self.cargar_pesos(n))
        else:
            self.modelo.set_weights(self.cargar_elite(n))

    def inferir_modelo(self, raycasts, speed):
        # Preprocesamiento
        r = np.array(raycasts)
        if speed > 0:
            v = np.array([speed])
        else:
            v = np.zeros(1)

        x = np.concatenate((r, v), axis=0)
        x = x.round(decimals=5)

        result = self.modelo.predict(x.reshape(1, 6), verbose=0)

        # Saca una imagen de la estructura del modelo
        # tf.keras.utils.plot_model(self.modelo, to_file="model_shape_info.png", show_dtype=True, show_shapes=True, show_layer_names=True)

        return result

    def guardar_pesos(self, n):
        # Guardar pesos Driver actual en archivo .npy
        weights = self.modelo.get_weights()
        chain = weights[0]
        for i in weights[1:]:
            chain = np.append(chain, i)
        np.save(f"weights/pesos{n}.npy", chain)

    @staticmethod
    def inicializar_pesos():
        # Generar pesos aleatorios para nuevo Driver, en la primera generación
        #w_init = tf.keras.initializers.GlorotUniform()
        w_init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        b_init = tf.keras.initializers.RandomNormal(mean=0., stddev=2.)
        w1 = w_init(shape=(6, 9))
        b1 = b_init(shape=(9, ))
        w2 = w_init(shape=(9, 6))
        b2 = b_init(shape=(6, ))
        w3 = w_init(shape=(6, 4))
        b3 = b_init(shape=(4, ))
        w4 = w_init(shape=(4, 1))
        b4 = b_init(shape=(1, ))
        return [w1, b1, w2, b2, w3, b3, w4, b4, w4, b4]

    @staticmethod
    def cargar_pesos(n):
        # Carga archivo .npy
        chain = np.load(f"weights/pesos{n}.npy", allow_pickle=True)
        w1 = chain[0:54].reshape((6, 9))
        b1 = chain[54:63].reshape((9,))
        w2 = chain[63:117].reshape((9, 6))
        b2 = chain[117:123].reshape((6,))
        w3 = chain[123:147].reshape((6, 4))
        b3 = chain[147:151].reshape((4,))
        w4_1 = chain[151:155].reshape((4, 1))
        b4_1 = chain[155].reshape((1,))
        w4_2 = chain[156:160].reshape((4, 1))
        b4_2 = chain[160].reshape((1,))
        return [w1, b1, w2, b2, w3, b3, w4_1, b4_1, w4_2, b4_2]

    @staticmethod
    def cargar_elite(n):
        # Carga mejores corredores de la carpeta élite .npy
        listado = os.listdir("tmp/elite")
        chain = np.load(f"tmp/elite/{listado[n]}", allow_pickle=True)
        w1 = chain[0:54].reshape((6, 9))
        b1 = chain[54:63].reshape((9,))
        w2 = chain[63:117].reshape((9, 6))
        b2 = chain[117:123].reshape((6,))
        w3 = chain[123:147].reshape((6, 4))
        b3 = chain[147:151].reshape((4,))
        w4_1 = chain[151:155].reshape((4, 1))
        b4_1 = chain[155].reshape((1,))
        w4_2 = chain[156:160].reshape((4, 1))
        b4_2 = chain[160].reshape((1,))
        return [w1, b1, w2, b2, w3, b3, w4_1, b4_1, w4_2, b4_2]
