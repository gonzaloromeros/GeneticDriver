#! /usr/bin/env python3
from pytocl.main import main
from my_driver import MyDriver
import numpy as np
from os import remove
import os
from modelo import Modelo
import random
import matplotlib
import genetic
import matplotlib.pyplot as plt


if __name__ == '__main__':

    train = False
    if train:
        # Aplica el Algoritmos Genéticos para conseguir un conductor capaz de recorrer el circuito
        generaciones = 351  # Generaciones +1

        # Drivers por cada tipo de selección (población)
        elite = 4
        nuevos = 10
        n_padres = 26
        # Población de cada generación
        poblacion = int(elite + nuevos + n_padres*2)

        if os.path.isfile("fitness/fitness_graph.txt"):
            remove("fitness/fitness_graph.txt")

        for g in range(1, generaciones):
            print(f'--Generation {g}--')
            # Comprueba que no haya un fichero fitness a medias
            if os.path.isfile("fitness/fitness.npy"):
                remove("fitness/fitness.npy")

            for i in range(0, poblacion):
                repite = True
                while repite:
                    repite = False
                    print(f'Driver {i+1}:')
                    main(MyDriver(logdata=False, generation=g, n=i))
                    '''
                    if g > 1:
                        # Cada Driver hace otro intento para asegurar que los resultados sean confiables
                        main(MyDriver(logdata=False, generation=g, n=i))
                        # Calcula media del fitness del Driver
                        fitness = np.load("fitness/fitness.npy", allow_pickle=True)
                        fit = np.average(fitness[-2:])
                        if fit > 0:
                            fitness = np.concatenate((fitness[:-2], [fit]))
                            print(fit)
                        else:
                            fitness = fitness[:-2]
                            repite = True
                            pesos_init = Modelo.inicializar_pesos()
                            pesos = np.concatenate((pesos_init[0], pesos_init[1]), axis=None)
                            for idx, peso in enumerate(pesos_init):
                                if idx >= 2:
                                    pesos = np.concatenate((pesos, pesos_init[idx]), axis=None)
                            np.save(f"weights/pesos{i}.npy", pesos)
                        np.save(f"fitness/fitness.npy", fitness)
                    '''
                print('-------------------------------')

            # Carga fitness
            fitness = np.load("fitness/fitness.npy", allow_pickle=True)
            # Salva los fitness con el nombre de la generación actual
            np.save(f"fitness/fitness{g}.npy", fitness)
            remove("fitness/fitness.npy")

            # Ordenar por peso de mayor a menor
            index_orden = np.argsort(-fitness)

            # Escribe en un fichero de texto los datos para poder graficarlo
            graph_file = open("fitness/fitness_graph.txt", 'a')
            graph_file.write(f'{g},{"{:.2f}".format(np.average(fitness))},{"{:.2f}".format(fitness[index_orden[0]])}\n')

            # Selección nueva generación --> 80 individuos
            # - Stochastic universal sampling (SUS) (n_padres x 2 hijos)
            padres = genetic.sus(fitness, index_orden, n_padres)
            # Limpia carpeta antes de añadir pesos
            for i in range(0, poblacion):
                if os.path.isfile(f"tmp/parents/pesos{i}.npy"):
                    remove(f"tmp/parents/pesos{i}.npy")
            # Copia pesos de padres correspondientes
            for driver in padres:
                genetic.copiar_pesos(driver, 'parents')
            # -- Devuelve un array emparejando el indice con su valor
            emparejamientos = np.random.permutation(np.arange(len(padres)))
            # *

            # - Elitismo
            # Limpia carpeta antes de añadir pesos
            for i in range(0, poblacion):
                if os.path.isfile(f"tmp/elite/pesos{i}.npy"):
                    remove(f"tmp/elite/pesos{i}.npy")
            # Copia los pesos con fitness más altos en la siguiente generación
            for i in range(0, elite):
                genetic.copiar_pesos(index_orden[i], 'elite')
            # -- Se sustituyen los 4 primeros pesos por los mejores Drivers de la generación anterior
            for i in range(0, elite):
                pesos = np.load(f"tmp/elite/pesos{index_orden[i]}.npy", allow_pickle=True)
                np.save(f"weights/pesos{i}.npy", pesos)

            # - Completamente nuevos
            for i in range(elite, elite+nuevos):
                pesos_init = Modelo.inicializar_pesos()
                pesos = np.concatenate((pesos_init[0], pesos_init[1]), axis=None)
                for idx, peso in enumerate(pesos_init):
                    if idx >= 2:
                        pesos = np.concatenate((pesos, pesos_init[idx]), axis=None)
                np.save(f"weights/pesos{i}.npy", pesos)

            # *
            # - Termina el emparejamiento de los padres
            genetic.crossover_multipunto(emparejamientos, padres, elite+nuevos)

            # Mutaciones
            for i in range(elite, poblacion):
                if 6 > random.randint(0, 100):
                    genetic.mutacion(i)

    else:
        # Grafica los datos recogidos (rojo mejor Driver) (verde media de fitness)
        datos = open("fitness/fitness_graph.txt", 'r').read()
        lineas = datos.split('\n')
        xs = []
        ys = []
        bs = []
        for linea in lineas:
            if len(linea) > 1:
                x, y, b = linea.split(',')
                xs.append(int(x))
                ys.append(float(y))
                bs.append(float(b))

        plt.plot(xs, bs, '-r')
        plt.plot(xs, ys, '-g')
        plt.show()

        # Probar con los 4 Drivers que haya conseguido más fitness en el entrenamiento
        for i in range(0, 4):
            print(f'Driver {i + 1}:')
            main(MyDriver(logdata=False, generation=-1, n=i))

            print('-------------------------------')


