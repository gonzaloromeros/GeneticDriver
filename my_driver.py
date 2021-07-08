from pytocl.driver import Driver
from pytocl.car import State, Command
import numpy as np
import genetic
import math


class MyDriver(Driver):

    def drive(self, carstate: State) -> Command:
        command = Command()

        raycasts = np.array(carstate.distances_from_edge)[[0, 3, 9, 15, 18]]
        #if abs(carstate.angle) < 15:
        #    self.distance_raced = carstate.distance_raced # TODO: uyeuude
        #self.distance_raced = carstate.distance_raced

        if self.time < math.trunc(carstate.current_lap_time) and np.average(raycasts) != 1:
            # Ajusta tiempo
            #self.time += 0.08
            self.time += carstate.current_lap_time
            # Predicción del modelo
            #self.prediction = self.modelo.inferir_modelo(raycasts, carstate.speed_x)
            #TODO: modelo cambiado para que la predicción sea en cierto momento del tiempooooooooooooooooooooooooooooooooooooo
            #self.prediction = self.modelo.inferir_modelo(raycasts)

        #TODO: afafgawg
        # Cada medio segundo se hace la predicción
        # Control de precisión de datos, los raycasts tardan 1 segundo en volver a mandar información precisa
        # Ajusta tiempo
        # Contador de vueltas
        if math.trunc(self.time) < math.trunc(carstate.current_lap_time) or\
                (self.angle_av[1] == 0 and carstate.current_lap_time > 0):
            if (1.0 > carstate.current_lap_time > 0.0) and carstate.last_lap_time != 0.0:
                self.lap += 1
                print(f"lap: {self.lap}")

        if self.time_pred < carstate.current_lap_time or \
                (self.angle_av[1] == 0 and carstate.current_lap_time > 0):
            self.prediction = self.modelo.inferir_modelo(raycasts)
            # Añadimos los segundos necesarios
            self.time_pred += 0.2
            #TODO: distancia (d)
            self.distance_raced = carstate.distance_raced
            #if abs(carstate.angle) < 30:
                ##########################self.distance_raced = carstate.distance_raced
            self.raycast_av[int(self.angle_av[1] % 15)] = (np.average(raycasts))
            self.speed_av[int(self.angle_av[1] % 15)] = carstate.speed_x

            # Preprocesamiento para fitness
            ######################################################self.speed_av[0] += carstate.speed_x
            #self.raycast_av[0] += (np.average(raycasts) + raycasts[2])/2  # TODO: raycasts[2]
            #######self.raycast_av[0] += np.average(raycasts)
            #self.raycast_av[0] += (raycasts[1]+raycasts[2]+raycasts[3])/3
            #self.raycast_av[0] += (raycasts[1]+raycasts[3])/2  # TODO: doble 45º
            self.equidistance_av[0] += abs(carstate.distance_from_center)
            self.angle_av[0] += abs(carstate.angle)
            ########self.speed_av[1] += 1
            ######self.raycast_av[1] += 1
            self.equidistance_av[1] += 1
            self.angle_av[1] += 1

        command.steering = self.prediction[0, 0]
        #TODO: ada
        '''----------------'''
        if self.prediction[0, 1] >= 0:
            # Se divide entre 2 para que el acelerador no lo pueda llevar al máximo y evitar altas velocidades
            command.accelerator = (self.prediction[0, 1])/1.4  #TODO: /1.2
            command.brake = 0
        else:
            command.brake = abs(self.prediction[0, 1])/1.6
            command.accelerator = 0
        '''--------- '''
        #command.accelerator = 0.4  # TODO: aweda
        #command.brake = 0

        command.gear = self.caja_cambios(carstate.rpm, carstate.gear, self.time, self.max_gear)

        if carstate.distance_from_center < -1.1 or 1.1 < carstate.distance_from_center or \
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

        #TODO: printeo de pesos
        weights = self.modelo.modelo.get_weights()
        chain = weights[0]
        for i in weights[1:]:
            chain = np.append(chain, i)
        print(chain)

        # Preprocesamiento para fitness
        speed_av = np.average(self.speed_av)
        raycast_av = np.average(self.raycast_av)
        for idx, v in enumerate(self.speed_av):
            if v == 0 and speed_av == np.average(self.speed_av):
                speed_av = np.average(self.speed_av[:idx-1])
                raycast_av = np.average(self.raycast_av[:idx-1])

        equidistance_av = self.equidistance_av[0]/self.equidistance_av[1]
        angle_av = self.angle_av[0]/self.angle_av[1]

        if np.isnan(speed_av):
            speed_av = 0

        # Función fitness
        fit = genetic.fitness_func(self.distance_raced, speed_av, self.lap, raycast_av, equidistance_av, angle_av, self.time)
        if self.generation <= 1:
            print(fit)
        genetic.guardar_fitness(fit)


