#! /usr/bin/env python3
from pytocl.main import main
from my_driver import MyDriver
import numpy as np
from os import remove
import genetic


if __name__ == '__main__':

    train = True

    if train:
        # Aplica el Algoritmos Genéticos para conseguir un conductor capaz de recorrer el circuito
        for g in range(1, 11):
            print(f'--Generation {g}--')

            for i in range(1, 100):
                print(f'Driver {i}:')
                main(MyDriver(logdata=False, generation=g, n=i))

            # Selección nueva generación --> 80 individuos
            fitness = np.load("fitness/fitness.npy")

            index_orden = np.zeros_like(fitness)
            indexes = np.argsort(fitness)
            for i in range(len(indexes)):
                index_orden[i] = indexes[(len(indexes) - 1) - i]

            # - Stochastic universal sampling (SUS) 85% --> 68
            padres = genetic.sus(fitness, index_orden)

            # - Elitismo 5% --> 4
            # - Completamente nuevos 10% --> 8

            #
            np.save(f"fitness/fitness{g}.npy", fitness)
            remove("fitness/fitness.npy")

    else:
        # Probar con el Driver que haya conseguido más fitness en el entrenamiento
        ...