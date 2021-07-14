import numpy as np
import os
import random


def fitness_func(d, v, l, r, p, a, t):
    # d = distancia recorrida (distance raced)
    # v = velocidad media (average speed)
    # l = vueltas (laps)
    # r = media de la media de los raycasts (average raycasts measures)
    # p = posición media con respecto al centro de la pista [0 = centro] (average position of the car from the center)
    # a = angulo medio con respecto al eje de la pista [0 = paralelo] (average angle of the car from the center axis)
    # t = tiempo en segundos (time in seconds)
    print(f'(d: {"{:.2f}".format(d)} | v: {"{:.2f}".format(v)} | l: {"{:.2f}".format(l)} | r: {"{:.2f}".format(r)} | p: {"{:.2f}".format(p)} | a: {"{:.2f}".format(a)} | t: {"{:.2f}".format(t)})')

    # Función fitness, sustituir si se quiere modificar
    fit = (np.trunc(d/2) * l * r) / (1 + r)

    if fit < 0:
        fit = 0.0
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
    # ordena el array fitness y lo redondea a 3 decimales
    fitness_orden = np.zeros_like(fitness)
    for idx, fit in enumerate(fitness_orden):
        fitness_orden[idx] = round(fitness[index_orden[idx]], 3)
    # iterador para controlar en que parte de los individuos estamos
    j = 0
    # sumatorio de los fitness para compararlos con la posición de los punteros
    sum_fitness = fitness_orden[j]
    # array donde se guardará la posición que ocupan los padres dentro del array de fitness
    padres = np.full(n_padres, -1, dtype=int)
    # intervalo entre punteros
    intervalo = total_fitness / n_padres
    # posición inicial donde empezará el primer puntero
    puntero = np.random.uniform(0., intervalo-0.001)

    # Hace la selección de los padres aplicando el SUS, limitando que un padre no puede ser seleccionado más de 2 veces
    duplicado = False
    for i in range(n_padres):
        sig = False
        while not sig:  # Hasta conseguir un padre
            if puntero < sum_fitness:
                if not duplicado:
                    padres[i] = index_orden[j]
                    if i != 0 and padres[i] == padres[i-1]:
                        duplicado = True
                puntero += intervalo
                sig = True
            else:
                j += 1
                sum_fitness += fitness_orden[j]
                duplicado = False
    # Rellena con los mejores individuos si algún padre se ha quedado con -1 por haber saturado la selección anterior
    j = 0
    relleno = False
    for idx, padre in enumerate(padres):
        while not relleno and j < len(index_orden):
            usado = False
            for val in padres:
                if val == index_orden[j] and not usado:
                    j += 1
                    usado = True
            if not usado:
                relleno = True

        if padre == -1:
            padres[idx] = index_orden[j]
            relleno = False
    return padres


def copiar_pesos(driver, carpeta):
    peso = np.load(f"weights/pesos{driver}.npy", allow_pickle=True)
    np.save(f"tmp/{carpeta}/pesos{driver}.npy", peso)


def crossover_simple(emparejamientos, padres, driver):
    for idx, val in enumerate(emparejamientos):
        if idx != val:
            # Carga los pesos de los padres
            padre1 = np.load(f"tmp/parents/pesos{padres[idx]}.npy", allow_pickle=True)
            padre2 = np.load(f"tmp/parents/pesos{padres[val]}.npy", allow_pickle=True)
            # Inicializa a los hijos
            hijo1 = np.zeros_like(padre1)
            hijo2 = np.zeros_like(padre2)

            # Selecciona varios segmentos a intercambiar
            x = random.randint(0, len(padre1) - 1)

            # Asigna los pesos a los hijos
            for i in range(0, len(padre1)):
                if i >= x:
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
        else:
            mutacion(driver)
            driver += 1
            mutacion(driver)
            driver += 1


def crossover_multipunto(emparejamientos, padres, driver):
    # Genera descendencia de los emparejamientos dados intercambiando 4 cadenas de genes
    for idx, val in enumerate(emparejamientos):
        if idx != val:
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
        else:
            mutacion(driver)
            driver += 1
            mutacion(driver)
            driver += 1


def crossover_genes(emparejamientos, padres, driver):
    # Genera descendencia de los emparejamientos dados intercambiando genes unitariamente
    probabilidad_cambio_gen = 50
    for idx, val in enumerate(emparejamientos):
        if idx != val:
            # Carga los pesos de los padres
            padre1 = np.load(f"tmp/parents/pesos{padres[idx]}.npy", allow_pickle=True)
            padre2 = np.load(f"tmp/parents/pesos{padres[val]}.npy", allow_pickle=True)
            # Inicializa a los hijos
            hijo1 = np.zeros_like(padre1)
            hijo2 = np.zeros_like(padre2)

            for i in range(0, len(padre1)):
                # Intercambia gen por gen con un 50% de probabilidad
                if random.randint(1, 100) <= probabilidad_cambio_gen:
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
        else:
            mutacion(driver)
            driver += 1
            mutacion(driver)
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
    while np.sum(paring) > 2:
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
    prob_sig_mutacion = 0.7
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
