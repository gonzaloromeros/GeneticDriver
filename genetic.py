import numpy as np
import os
import random
import tensorflow as tf


def fitness_func(d, v, l, r, p, a, t):
    # d = distance raced
    # v = average speed
    # l = laps
    # r = average middle raycast measures
    # p = average position of the car from the center
    # a = average angle of the car from the center axis
    # t = tiempo con precisión de 0.5s
    print(f'(d: {"{:.2f}".format(d)} | v: {"{:.2f}".format(v)} | l: {"{:.2f}".format(l)} | r: {"{:.2f}".format(r)} | p: {"{:.2f}".format(p)} | a: {"{:.2f}".format(a)} | t: {"{:.2f}".format(t)})')

    #fit = (d * l * r) / ((1 + r) * ((1+a)**2) * ((1+p)**3))
    #fit = ( (d * l * r) / ((1 + r) * (1+a)) )

    #fit = ( ((d**2 + t) * l) / (1 + p + a) )
    #fit = (((d + t) * l) / (1 + a))
    #fit = ( ((d + t) * l * r) / ((1 + r) * (1 + p)) )
    #fit = ((d + v) * l * r) / (((1+a)**2) * ((1+p)**3))

    #fit = ((d + v) * l * r) / (1 + a)

    #fit = (d**2 + v + r - p - a + t) / 2  # 102m Brn
    #fit = ((d*2+t)*2)/(p+a) # 89m Gnz
    #fit = (d * 0.4) + ((l * 42) / (t * 0.2))  # 15m Jv
    #fit = ((d * 0.4) + (l * 42)) / (t * 0.2)  # 15m Jv
    fit = (((d+v)+((t*4)/(p+a)))*r)*0.1*l  # 158m Gnz

    print(fit)
    return fit


def guardar_fitness(fit):
    # Guarda el "fitness score" cada Driver en un archivo .npy
    if os.path.isfile("fitness/fitness.npy"):
        fitness = np.load("fitness/fitness.npy", allow_pickle=True)
        fitness = np.append(fitness, fit)
    else:
        fitness = np.array([fit])
    np.save("fitness/fitness.npy", fitness)


def sus(fitness, index_orden, n_padres):
    # Función que seleccióna a los padres de la siguiente generación siguiendo el "Muestreo universal estocástico" (SUS)
    # suma total de todos los valores del fitness
    total_fitness = np.sum(fitness)
    # iterador para controlar en que parte de los individuos estamos
    j = 0
    # sumatorio de los fitness para compararlos con la posición de los punteros
    sum_fitness = fitness[index_orden[j]]
    # número de punteros y por tanto de padres que se elegirán
    n_punteros = n_padres
    # array donde se guardará la posición que ocupan los padres dentro del array de fitness
    padres = np.zeros(n_punteros, dtype=int)
    # intervalo entre punteros
    intervalo = total_fitness / n_punteros
    # posición inicial donde empezará el primer puntero
    puntero = np.random.uniform(0., intervalo-0.01)

    for i in range(n_punteros):
        sig = False
        while not sig:
            if puntero < sum_fitness:
                padres[i] = index_orden[j]
                puntero += intervalo
                sig = True
            else:
                j += 1
                sum_fitness += fitness[index_orden[j]]
    return padres


def copiar_pesos(driver, carpeta):
    peso = np.load(f"weights/pesos{driver}.npy", allow_pickle=True)
    np.save(f"tmp/{carpeta}/pesos{driver}.npy", peso)


def crossover_multipunto(emparejamientos, padres, driver):
    # Genera toda la descendencia de los emparejamientos dados
    for idx, val in enumerate(emparejamientos):
        # Carga los pesos de los padres
        padre1 = np.load(f"tmp/parents/pesos{padres[idx]}.npy", allow_pickle=True)
        padre2 = np.load(f"tmp/parents/pesos{padres[val]}.npy", allow_pickle=True)
        # Inicializa a los hijos
        hijo1 = np.zeros_like(padre1)
        hijo2 = np.zeros_like(padre2)

        # Selecciona varios segmentos a intercambiar
        x1 = random.randint(0, int(len(padre1)/2)-1)
        x3 = random.randint(int(len(padre1)/2+1), len(padre1))
        x2 = random.randint(x1, x3)
        paring = np.random.randint(2, size=4)

        # Asigna los pesos a los hijos
        for i in range(0, len(padre1)):
            if (0 < i <= x1 and paring[0] == 1) or (x1 < i <= x2 and paring[1] == 1) or \
               (x2 < i <= x3 and paring[2] == 1) or (x3 < i < len(padre1) and paring[3] == 1):
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


def mezclar_pesos_inicio(pesos, poblacion):
    # Mezcla pesos aleatorios con pesos de la anterior generación
    n_driver = random.randint(0, poblacion - 1)
    peso_anterior = np.load(f"weights/pesos{n_driver}.npy", allow_pickle=True)

    # Selecciona los segmentos a intercambiar
    x1 = random.randint(0, int(len(pesos) / 2) - 1)
    x3 = random.randint(int(len(pesos) / 2 + 1), len(pesos))
    x2 = random.randint(x1, x3)
    # Array binario que decide si el segmento se intercambia (1) o se deja igual (0)
    paring = np.random.randint(2, size=4)

    # Modifica los pesos para intentar ganar competencia
    for i in range(0, len(pesos)):
        if (0 < i <= x1 and paring[0] == 1) or (x1 < i <= x2 and paring[1] == 1) or \
                (x2 < i <= x3 and paring[2] == 1) or (x3 < i < len(pesos) and paring[3] == 1):
            pesos[i] = peso_anterior[i]

    return pesos


def mutacion(driver):
    # Hace que los pesos de un Driver cambien al menos una vez
    probabilidad_mutacion = 100
    prob_sig_mutacion = 0.9
    mutando = True
    # Carga pesos a mutar
    pesos = np.load(f"weights/pesos{driver}.npy", allow_pickle=True)
    # Bucle de mutaciones con cada vez menor probabilidad de continuar
    while mutando:
        if random.randint(0, 100) <= probabilidad_mutacion:
            # Escoge un gen aleatorio
            gen = random.randint(0, len(pesos)-1)
            # Genera mutación
            mut_change = random.uniform(-0.5, 0.5)
            mut = gen + mut_change
            # Aplica mutación
            pesos[gen] = mut
            # Baja la probabilidad de otra mutación consecutiva
            probabilidad_mutacion = probabilidad_mutacion*prob_sig_mutacion
        else:
            # Termina la mutación
            mutando = False
    # Guarda los pesos mutados
    np.save(f"weights/pesos{driver}.npy", pesos)
