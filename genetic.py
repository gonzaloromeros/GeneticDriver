import numpy as np
import os
import random
import tensorflow as tf


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
    puntero = np.random.uniform(0., intervalo)

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


def copiar_pesos(driver, carpeta):
    peso = np.load(f"weights/pesos{driver}.npy")
    np.save(f"tmp/{carpeta}/pesos{driver}.npy", peso)


def crossover_multipunto(emparejamientos, padres, driver):
    # Genera toda la descendencia de los emparejamientos dados
    for idx, val in enumerate(emparejamientos):
        # Carga los pesos de los padres
        padre1 = np.load(f"tmp/parents/pesos{padres[idx]}.npy")
        padre2 = np.load(f"tmp/parents/pesos{padres[val]}.npy")
        # Selecciona un segmento a intercambiar
        x = random.randint(0, len(padre1)-1)
        y = random.randint(x, len(padre1))
        # Inicializa a los hijos
        hijo1 = np.zeros_like(padre1)
        hijo2 = np.zeros_like(padre2)
        # Asigna los pesos a los hijos
        for i in range(0, len(padre1)):
            if x < i <= y:
                hijo1[i] = padre2[i]
                hijo2[i] = padre1[i]
            else:
                hijo1[i] = padre1[i]
                hijo2[i] = padre2[i]
        # Guarda los pesos de los hijos
        np.save(f"weights/pesos{driver}.npy", hijo1)
        driver += 1
        np.save(f"weights/pesos{driver}.npy", hijo2)
        driver += 1


def mutacion(driver):
    # Hace que los pesos de un driver cambien al menos una vez
    probabilidad_mutacion = 100
    prob_sig_mutacion = 0.6
    mutando = True
    # Carga pesos a mutar
    pesos = np.load(f"weights/pesos{driver}.npy")
    # Bucle de mutaciones con cada vez menor probabilidad de continuar
    while mutando:
        if random.randint(1, 101) <= probabilidad_mutacion:
            # Escoge un gen aleatorio
            gen = random.randint(0, len(pesos))
            # Genera mutación
            mut_init = tf.keras.initializers.GlorotUniform()
            mut = mut_init(shape=(1,))
            # Aplica mutación
            pesos[gen] = mut
            # Baja la probabilidad de otra mutación consecutiva
            probabilidad_mutacion = probabilidad_mutacion*prob_sig_mutacion
        else:
            # Termina la mutación
            mutando = False
    # Guarda los pesos mutados
    np.save(f"weights/pesos{driver}.npy", pesos)
