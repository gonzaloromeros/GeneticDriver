#! /usr/bin/env python3
from pytocl.main import main
from my_driver import MyDriver
import numpy as np
from os import remove
from modelo import Modelo
import random
import genetic


if __name__ == '__main__':

    train = True
    if train:
        # Aplica el Algoritmos Genéticos para conseguir un conductor capaz de recorrer el circuito
        generaciones = 11
        poblacion = 81
        for g in range(1, generaciones):
            print(f'--Generation {g}--')

            for i in range(1, poblacion):
                print(f'Driver {i}:')
                main(MyDriver(logdata=False, generation=g, n=i))

            # Selección nueva generación --> 80 individuos
            fitness = np.load("fitness/fitness.npy")

            index_orden = np.zeros_like(fitness)
            indexes = np.argsort(fitness)
            for i in range(len(indexes)):
                index_orden[i] = indexes[(len(indexes) - 1) - i]

            # - Stochastic universal sampling (SUS) 85% --> 68 (34 padres)
            padres = genetic.sus(fitness, index_orden)
            for driver in padres:
                genetic.copiar_pesos(driver, 'parents')
            # -- Devuelve un array emparejando el indice con su valor
            emparejamientos = np.random.permutation(padres)
            # *
            # - Elitismo 5% --> 4
            elite = 4
            for i in range(0, elite):
                genetic.copiar_pesos(index_orden[i], 'elite')
            # -- Se sustituyen los 4 primeros pesos por los mejores Drivers de la generación anterior
            for i in range(0, elite):
                pesos = np.load(f"tmp/elite/pesos{index_orden[i]}.npy")
                np.save(f"weights/pesos{i+1}.npy", pesos)
            # - Completamente nuevos 10% --> 8
            nuevos = 8
            for i in range(elite, elite+nuevos):
                pesos = Modelo.inicializar_pesos()
                np.save(f"weights/pesos{i+1}.npy", pesos)
            # *
            # - Termina el emparejamiento de los padres
            genetic.crossover_multipunto(emparejamientos, padres, elite+nuevos+1)

            # Mutaciones
            for i in range(1, poblacion):
                if 5 > random.randint(0, 100):
                    genetic.mutacion(i)

            # Salva los fitness con el nombre de la generación actual
            np.save(f"fitness/fitness{g}.npy", fitness)
            remove("fitness/fitness.npy")

    else:
        # Probar con el Driver que haya conseguido más fitness en el entrenamiento
        ...