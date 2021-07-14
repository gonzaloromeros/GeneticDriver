from pytocl.driver import Driver
from pytocl.car import State, Command
import numpy as np
import genetic
import math


class MyDriver(Driver):

    def drive(self, carstate: State) -> Command:
        command = Command()

        raycasts = np.array(carstate.distances_from_edge)[[0, 3, 9, 15, 18]]

        if self.time < math.trunc(carstate.current_lap_time) and np.average(raycasts) != 1:
            # Ajusta tiempo cada vez que se actualice el enviado por el servidor
            self.time = carstate.current_lap_time
            # Predicción del modelo si se quiere hacer a cada 'tick' del juego
            # self.prediction = self.modelo.inferir_modelo(raycasts)

        # Contador de vueltas
        if self.last_lap != carstate.last_lap_time:
            self.lap += 1
            print(f"lap: {self.lap}")
            self.last_lap = carstate.last_lap_time

        if self.time_pred < carstate.current_lap_time or \
                (self.angle_av[1] == 0 and carstate.current_lap_time > 0):
            # Añadimos los segundos que tardará en realizarse de nuevo este bloque de acciones entero.
            #   Si no se toca el 'timeout' del servidor, el tiempo que es óptimo para que no surjan excepciones
            #   empieza a partir de los 0.25 segundos.
            self.time_pred += 0.05

            # Hacemos la predicción del modelo
            self.prediction = self.modelo.inferir_modelo(raycasts)

            # Preprocesamiento de variables para fitness
            # Coge la distancia recorrida
            self.distance_raced = carstate.distance_raced
            # Comprueba que el coche no esté girado con respecto al eje de la pista, 30º de margen hacia cada lado
            #   if abs(carstate.angle) < 30:
            #       self.distance_raced = carstate.distance_raced

            modulo = int(self.angle_av[1] % 20)
            self.raycast_av[modulo] = (np.average(raycasts))
            self.speed_av[modulo] = carstate.speed_x
            self.equidistance_av[modulo] = abs(carstate.distance_from_center)
            self.angle_av[0] += abs(carstate.angle)
            self.angle_av[1] += 1

        # Crea los comandos de control, basándose en la predicción del modelo
        command.steering = self.prediction[0, 0]
        '''----------------'''
        if self.prediction[0, 1] >= 0:
            # Se divide entre 2 para que el acelerador no lo pueda llevar al máximo y evitar altas velocidades
            command.accelerator = (self.prediction[0, 1])/2.5  #TODO: /1.2
            command.brake = 0
        else:
            command.brake = abs(self.prediction[0, 1])/2
            command.accelerator = 0.1
        '''--------- '''

        # También se puede hacer la velocidad continua, acelerando a ritmo constante, si quiere obviarse esa parte del modelo
        #   command.accelerator = 0.3
        #   command.brake = 0

        # Función encargada de las marchas
        command.gear = self.caja_cambios(carstate.rpm, carstate.gear, self.time, self.max_gear)

        # Controla el reseteo del conductor
        if carstate.distance_from_center < -1.0 or 1.0 < carstate.distance_from_center or \
                (carstate.current_lap_time > 4 > carstate.speed_x) or np.average(raycasts) == -1:
            command.meta = 1

        return command

    @staticmethod
    def caja_cambios(rpm, gear, time, max_gear) -> int:
        # Controlar las marchas max permitido 6, pero lo capamos a 3 para que no aumente demasiado su velocidad
        if time < 1:
            gear = 1
        elif rpm >= 7500 and gear != max_gear:
            gear += 1
        elif rpm <= 4000 and gear != 1:
            gear -= 1
        return gear

    def on_restart(self):
        # Ejecución entre el cambio de Drivers
        self.modelo.guardar_pesos(self.n)

        # Posibilidad de imprimir los pesos por consola
        '''
        weights = self.modelo.modelo.get_weights()
        chain = weights[0]
        for i in weights[1:]:
            chain = np.append(chain, i)
        print(chain)
        '''

        # Preprocesamiento para fitness
        speed_av = np.average(self.speed_av)
        raycast_av = np.average(self.raycast_av)
        equidistance_av = np.average(self.equidistance_av)

        for idx, r in enumerate(self.raycast_av):
            if r == 0 and speed_av == np.average(self.raycast_av):
                raycast_av = np.average(self.raycast_av[:idx - 1])
                speed_av = np.average(self.speed_av[:idx-1])
                equidistance_av = np.average(self.equidistance_av[:idx - 1])
        if np.isnan(speed_av):
            speed_av = 0

        angle_av = self.angle_av[0]/self.angle_av[1]

        # Función fitness
        fit = genetic.fitness_func(self.distance_raced, speed_av, self.lap, raycast_av, equidistance_av, angle_av, self.time)
        if self.generation <= 1:
            print(fit)
        genetic.guardar_fitness(fit)


