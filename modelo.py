from keras.layers import Dense, concatenate, Input
from keras.models import Sequential, Model
import tensorflow as tf
import numpy as np
import os


class Modelo:

    def __init__(self, generation, n):
        # Capa de entrada
        layer1 = Input(shape=(5,))

        # Capas ocultas
        layer2 = Dense(5, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer=tf.keras.initializers.RandomUniform(minval=-1., maxval=1.))(layer1)
        layer3 = Dense(3, activation='sigmoid', use_bias=True, kernel_initializer='random_uniform', bias_initializer=tf.keras.initializers.RandomUniform(minval=-1., maxval=1.))(layer2)
        #TODO: bias init: RandomNormal

        # Capa de salida
        outputsteer = Dense(1, activation='tanh', use_bias=True, kernel_initializer='random_uniform', bias_initializer=tf.keras.initializers.RandomUniform(minval=-1., maxval=1.))(layer3)
        #outputspeed = Dense(1, activation='sigmoid', use_bias=True, bias_initializer='RandomNormal')(layer3)
        outputspeed = Dense(1, activation='tanh', use_bias=True, kernel_initializer='random_uniform', bias_initializer=tf.keras.initializers.RandomUniform(minval=-1., maxval=1.))(layer3)
        output = concatenate([outputsteer, outputspeed])

        # Crear arquitectura del modelo
        self.modelo = Model(inputs=layer1, outputs=output)

        if generation == 1:
            self.modelo.set_weights(self.inicializar_pesos())
        elif generation != -1:
            self.modelo.set_weights(self.cargar_pesos(n))
        else:
            self.modelo.set_weights(self.cargar_elite(n))

    def inferir_modelo(self, raycasts):
        #TODO: ojo con esto que también cambia al modificar las entradas del modelo
        # Preprocesamiento
        r = np.array(raycasts)
        x = r.round(decimals=2)

        # Predicción
        result = self.modelo.predict(x.reshape(1, 5), verbose=0)

        # Saca una imagen de la estructura del modelo
        #tf.keras.utils.plot_model(self.modelo, to_file="model_shape_info.png", show_dtype=True, show_shapes=True, show_layer_names=True)

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
        w_init = tf.keras.initializers.GlorotUniform()
        #TODO: b_init = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.5)
        #b_init = tf.keras.initializers.zeros()
        b_init = tf.keras.initializers.RandomUniform(minval=-1., maxval=1.)
        w1 = w_init(shape=(5, 5))
        b1 = b_init(shape=(5, ))
        w2 = w_init(shape=(5, 3))
        b2 = b_init(shape=(3, ))
        '''
        w3 = w_init(shape=(3, 1))
        b3 = b_init(shape=(1, ))
        return [w1, b1, w2, b2, w3, b3, w3, b3]
        '''
        w3_1 = w_init(shape=(3, 1))
        b3_1 = b_init(shape=(1,))
        w3_2 = w_init(shape=(3, 1))
        b3_2 = b_init(shape=(1,))
        return [w1, b1, w2, b2, w3_1, b3_1, w3_2, b3_2]

    @staticmethod
    def cargar_pesos(n):
        # Carga archivo .npy
        chain = np.load(f"weights/pesos{n}.npy", allow_pickle=True)
        w1 = chain[0:25].reshape((5, 5))
        b1 = chain[25:30].reshape((5,))
        w2 = chain[30:45].reshape((5, 3))
        b2 = chain[45:48].reshape((3,))
        w3_1 = chain[48:51].reshape((3, 1))
        b3_1 = chain[51].reshape((1,))
        w3_2 = chain[52:55].reshape((3, 1))
        b3_2 = chain[55].reshape((1,))
        return [w1, b1, w2, b2, w3_1, b3_1, w3_2, b3_2]
        '''
        w3_1 = chain[30:33].reshape((3, 1))
        b3_1 = chain[33].reshape((1,))
        w3_2 = chain[34:37].reshape((3, 1))
        b3_2 = chain[37].reshape((1,))
        return [w1, b1, w3_1, b3_1, w3_2, b3_2]
        '''

    @staticmethod
    def cargar_elite(n):
        # Carga mejores corredores de la carpeta élite .npy
        listado = os.listdir("tmp/elite")
        chain = np.load(f"tmp/elite/{listado[n]}", allow_pickle=True)
        w1 = chain[0:25].reshape((5, 5))
        b1 = chain[25:30].reshape((5,))
        w2 = chain[30:45].reshape((5, 3))
        b2 = chain[45:48].reshape((3,))
        w3_1 = chain[48:51].reshape((3, 1))
        b3_1 = chain[51].reshape((1,))
        w3_2 = chain[52:55].reshape((3, 1))
        b3_2 = chain[55].reshape((1,))
        return [w1, b1, w2, b2, w3_1, b3_1, w3_2, b3_2]
        '''
        w3_1 = chain[30:33].reshape((3, 1))
        b3_1 = chain[33].reshape((1,))
        w3_2 = chain[34:37].reshape((3, 1))
        b3_2 = chain[37].reshape((1,))
        return [w1, b1, w3_1, b3_1, w3_2, b3_2]
        '''
