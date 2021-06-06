import numpy as np
import os
import random


def fitness_func(d, v, l, r):
    # d = distance raced
    print(f'd: {d}')
    # v = average speed
    print(f'v: {v}')
    # l = laps
    print(f'l: {l}')
    # R = average raycast measures
    print(f'r: {r/(1+r)}')

    fit = ((d ** 2) * v * l * r) / (1 + r)
    return fit


def guardar_fitness(fit):
    # Guarda el "fitness score" cada Driver en un archivo .npy
    if os.path.isfile("fitness/fitness.npy"):
        fitness = np.load("fitness/fitness.npy")
        fitness = np.append(fitness, fit)
    else:
        fitness = np.array([fit])
    np.save("fitness/fitness.npy", fitness)


def sus(fitness, index_orden):
    # Función que seleccióna a los padres de la siguiente generación siguiendo el "Muestreo universal estocástico" (SUS)
    # suma total de todos los valores del fitness
    total_fitness = np.sum(fitness)
    # iterador para controlar en que parte de los individuos estamos
    j = 0
    # sumatorio de los fitness para compararlos con la posición de los punteros
    sum_fitness = fitness[index_orden[j]]
    # número de punteros y por tanto de padres que se elegirán
    n_punteros = 34
    # array donde se guardará la posición que ocupan los padres dentro del array de fitness
    padres = np.zeros(n_punteros)
    # intervalo entre punteros
    intervalo = total_fitness / n_punteros
    # posición inicial donde empezará el primer puntero
    puntero = random.randint(0, intervalo)

    for i in range(n_punteros):
        sig = False
        while not sig:
            if puntero < sum_fitness:
                padres[i] = index_orden[j]
                sig = True
            else:
                j += 1
                sum_fitness += fitness[index_orden[j]]
    return padres
