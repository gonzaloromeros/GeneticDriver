from pytocl.driver import Driver
from pytocl.car import State, Command
import numpy as np
import genetic
import math


class MyDriver(Driver):

    def drive(self, carstate: State) -> Command:
        command = Command()

        raycasts = np.array(carstate.distances_from_edge)[[0, 3, 9, 15, 18]]

        self.distance_raced = carstate.distance_raced

        # Control de precisión de datos, los raycasts tardan 1 segundo en volver a mandar información precisa
        #TODO: afafgawg
        # Cada segundo se hace la predicción
        if self.time_pred < math.trunc(carstate.current_lap_time):
            # Ajusta tiempo
            self.time_pred = math.trunc(carstate.current_lap_time)
            print(raycasts)
            # Predicción del modelo
            self.prediction = self.modelo.inferir_modelo(raycasts, carstate.speed_x)

            # Preprocesamiento para fitness
            self.speed_av[0] += carstate.speed_x
            self.raycast_av[0] += np.average(raycasts)
            self.equidistance_av[0] += carstate.distance_from_center
            self.angle_av[0] += abs(carstate.angle)
            self.speed_av[1] += 1
            self.raycast_av[1] += 1
            self.equidistance_av[1] += 1
            self.angle_av[1] += 1

            # Contador de vueltas
            if (1.0 > carstate.current_lap_time > 0.0) and carstate.last_lap_time != 0.0:
                self.lap += 1
                print(f"lap: {self.lap}")

        command.steering = self.prediction[0, 0]
        if self.prediction[0, 1] >= 0:
            # Se divide entre 2 para que el acelerador no lo pueda llevar al máximo y evitar altas velocidades
            command.accelerator = (self.prediction[0, 1]) #TODO: /1.5
            command.brake = 0
        else:
            command.brake = abs(self.prediction[0, 1])
            command.accelerator = 0

        command.gear = self.caja_cambios(carstate.rpm, carstate.gear, carstate.current_lap_time)

        if carstate.distance_from_center < -1 or 1 < carstate.distance_from_center or \
                (carstate.current_lap_time > 3 > carstate.speed_x) or np.average(raycasts) == -1:
            command.meta = 1

        return command

    @staticmethod
    def caja_cambios(rpm, gear, time) -> int:
        # Controlar las marchas max permitido 6, pero lo capamos a 3 para que no aumente demasiado su velocidad
        if time < 1:
            gear = 1
        elif rpm >= 7500 and gear != 6:
            gear += 1
        elif rpm <= 4000 and gear != 1:
            gear -= 1
        return gear

    def on_restart(self):
        # Ejecución entre el cambio de Drivers
        self.modelo.guardar_pesos(self.n)

        # Preprocesamiento para fitness
        speed_av = self.speed_av[0]/self.speed_av[1]
        raycast_av = self.raycast_av[0]/self.raycast_av[1]
        equidistance_av = self.equidistance_av[0]/self.equidistance_av[1]
        angle_av = self.angle_av[0]/self.angle_av[1]

        # Función fitness
        fit = genetic.fitness_func(self.distance_raced, speed_av, self.lap, raycast_av, equidistance_av, angle_av)
        if self.generation <= 1:
            print(fit)
        genetic.guardar_fitness(fit)


