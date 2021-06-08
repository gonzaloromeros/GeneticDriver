#! /usr/bin/env python3
from pytocl.main import main
from my_driver import MyDriver
import numpy as np
from os import remove
import os
from modelo import Modelo
import random
import genetic


if __name__ == '__main__':

    train = True
    if train:
        # Aplica el Algoritmos Genéticos para conseguir un conductor capaz de recorrer el circuito
        generaciones = 51  # Generaciones +1
        poblacion = 80  # Población de cada generación
        for g in range(1, generaciones):
            print(f'--Generation {g}--')
            # Comprueba que no haya un fichero fitness a medias
            if os.path.isfile("fitness/fitness.npy"):
                remove("fitness/fitness.npy")

            for i in range(0, poblacion):
                print(f'Driver {i+1}:')
                main(MyDriver(logdata=False, generation=g, n=i))

            # Carga fitness
            fitness = np.load("fitness/fitness.npy", allow_pickle=True)
            # Salva los fitness con el nombre de la generación actual
            np.save(f"fitness/fitness{g}.npy", fitness)
            remove("fitness/fitness.npy")

            # Selección nueva generación --> 80 individuos
            # Ordenar por peso de mayor a menor
            index_orden = np.argsort(-fitness)
            # - Stochastic universal sampling (SUS) 85% --> 68 (34 padres)
            padres = genetic.sus(fitness, index_orden)
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

            # - Elitismo 5% --> 4
            elite = 4
            # Limpia carpeta antes de añadir pesos
            for i in range(0, poblacion):
                if os.path.isfile(f"tmp/elite/pesos{i}.npy"):
                    remove(f"tmp/elite/pesos{i}.npy")
            # Copia los 4 pesos con fitness más altos
            for i in range(0, elite):
                genetic.copiar_pesos(index_orden[i], 'elite')
            # -- Se sustituyen los 4 primeros pesos por los mejores Drivers de la generación anterior
            for i in range(0, elite):
                pesos = np.load(f"tmp/elite/pesos{index_orden[i]}.npy", allow_pickle=True)
                np.save(f"weights/pesos{i}.npy", pesos)

            # - Completamente nuevos 10% --> 8
            nuevos = 8
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
            for i in range(0, poblacion):
                if 5 > random.randint(0, 100):
                    genetic.mutacion(i)

    else:
        # Probar con los 4 Drivers que haya conseguido más fitness en el entrenamiento
        for i in range(0, 4):
            print(f'Driver {i + 1}:')
            main(MyDriver(logdata=False, generation=-1, n=i))
