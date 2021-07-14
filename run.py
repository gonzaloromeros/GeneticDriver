#! /usr/bin/env python3
from pytocl.main import main
from my_driver import MyDriver
import numpy as np
from os import remove
import os
from modelo import Modelo
import random
import genetic
import matplotlib.pyplot as plt


def grafica_elite():
    # Grafica los fitness recogidos en la ejecución
    #   (rojo - mejor Driver)
    #   (azul - media de los 2 siguientes mejores)
    #   (verde - media de fitness de toda la población)
    datos = open("fitness/fitness_graph.txt", 'r').read()
    lineas = datos.split('\n')
    xs = []
    ys = []
    bs = []
    b2s = []
    for linea in lineas:
        if len(linea) > 1:
            x, y, b, b2 = linea.split(',')
            xs.append(int(x))
            ys.append(float(y))
            bs.append(float(b))
            b2s.append(float(b2))

    plt.plot(xs, b2s, '-b')
    plt.plot(xs, bs, '-r')
    plt.plot(xs, ys, '-g')
    plt.show()


# __Inicio de programa__
if __name__ == '__main__':

    # Entrena o Infiere
    train = False

    # Drivers por cada tipo de selección (población)
    elite = 5
    nuevos = 8
    n_padres = 15

    if train:
        # Aplica Algoritmos Genéticos para conseguir un conductor capaz de recorrer el circuito
        generaciones = 301  # Generaciones +1

        # Población de cada generación
        poblacion = int(elite + nuevos + n_padres*4)

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
                    main(MyDriver(logdata=False, generation=g, n=i, max_gear=1))
                    # Comentar o descomentar sección para que cada Driver haga otro intento
                    #   para asegurarse de que los resultados son más confiables
                    '''----------------------------------------------------------------------------------------
                    if g > 1:
                        main(MyDriver(logdata=False, generation=g, n=i, max_gear=3))
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
                    ----------------------------------------------------------------------------------------- '''
                print('-------------------------------')

            # Carga fitness
            fitness = np.load("fitness/fitness.npy", allow_pickle=True)
            # Salva los fitness con el nombre de la generación actual
            np.save(f"fitness/fitness{g}.npy", fitness)
            remove("fitness/fitness.npy")

            # Ordenar por peso de mayor a menor
            index_orden = np.argsort(-fitness)

            # Creación de siguiente generación
            # Selección de los padres
            # - Stochastic universal sampling (SUS)
            padres = genetic.sus(fitness, index_orden, n_padres)

            # Limpia carpeta antes de añadir pesos
            for i in range(0, poblacion):
                if os.path.isfile(f"tmp/parents/pesos{i}.npy"):
                    remove(f"tmp/parents/pesos{i}.npy")
            # Copia pesos de padres correspondientes
            for driver in padres:
                genetic.copiar_pesos(driver, 'parents')

            # - Elitismo
            # Limpia carpeta antes de añadir pesos
            for i in range(0, poblacion):
                if os.path.isfile(f"tmp/elite/pesos{i}.npy"):
                    remove(f"tmp/elite/pesos{i}.npy")
            # Copia los pesos con fitness más altos en la siguiente generación
            for i in range(0, elite):
                genetic.copiar_pesos(index_orden[i], 'elite')
            # Se sustituyen los 4 primeros pesos por los mejores Drivers de la generación anterior
            for i in range(0, elite):
                pesos = np.load(f"tmp/elite/pesos{index_orden[i]}.npy", allow_pickle=True)
                np.save(f"weights/pesos{i}.npy", pesos)

            # - Completamente nuevos
            init_nuevos = np.trunc(nuevos / 4)
            for i in range(elite, elite + nuevos):
                pesos_init = Modelo.inicializar_pesos()
                pesos = np.concatenate((pesos_init[0], pesos_init[1]), axis=None)
                for idx, peso in enumerate(pesos_init):
                    if idx >= 2:
                        pesos = np.concatenate((pesos, pesos_init[idx]), axis=None)
                if init_nuevos > 0:
                    init_nuevos -= 1
                else:
                    pesos = genetic.mezclar_pesos_inicio(pesos, poblacion)
                np.save(f"weights/pesos{i}.npy", pesos)

            # Emparejamiento de los padres y generación de su descendencia
            # Devuelve un array emparejando el indice con su valor
            emparejamientos1 = np.random.permutation(np.arange(len(padres)))
            emparejamientos2 = np.random.permutation(np.arange(len(padres)))
            emparejamientos3 = np.random.permutation(np.arange(len(padres)))

            # Cada emparejamiento crearán: el número de padres x2, hijos.
            #   Hay que tenerlo en cuenta al definir tanto la población,
            #    como el último parámetro de las funciones de emparejamiento.
            n_hijos = len(emparejamientos1)

            # Funciones de emparejamiento
            # genetic.crossover_simple(emparejamientos1, padres, elite + nuevos)
            genetic.crossover_multipunto(emparejamientos2, padres, elite + nuevos)
            genetic.crossover_genes(emparejamientos3, padres, elite + nuevos + n_hijos*2)

            # Mutaciones
            for i in range(elite+1, poblacion):
                if 15 > random.randint(0, 100):
                    genetic.mutacion(i)

            # Escribe en un fichero de texto los datos para poder graficarlo
            graph_file = open("fitness/fitness_graph.txt", 'a')
            graph_file.write(f'{g},{"{:.2f}".format(np.average(fitness))},{"{:.2f}".format(fitness[index_orden[0]])},{"{:.2f}".format((fitness[index_orden[1]]+fitness[index_orden[2]])/2)}\n')

    else:
        # Grafica fitness de la elite y la media de la población
        grafica_elite()

        # Probar con los Drivers que haya conseguido más fitness en el entrenamiento
        for i in range(0, elite):
            print(f'Driver {i + 1}:')
            main(MyDriver(logdata=False, generation=-1, n=i, max_gear=1))

            print('-------------------------------')