from pytocl.driver import Driver
from pytocl.car import State, Command
import numpy as np
import genetic


class MyDriver(Driver):

    def drive(self, carstate: State) -> Command:
        command = Command()

        raycasts = np.array(carstate.focused_distances_from_edge)
        self.distance_raced = carstate.distance_raced

        # Control de precisión de datos, los raycasts tardan 1 segundo en volver a mandar información precisa
        if np.average(raycasts) != -1:
            # Predicción del modelo
            self.prediction = self.modelo.inferir_modelo(raycasts, carstate.speed_x)

            # Preprocesamiento para fitness
            self.speed_av[0] += carstate.speed_x
            self.raycast_av[0] += np.average(raycasts)
            self.speed_av[1] += 1
            self.raycast_av[1] += 1

            # Contador de vueltas
            if (1.0 > carstate.current_lap_time > 0.0) and carstate.last_lap_time != 0.0:
                self.lap += 1
                print(f"lap: {self.lap}")

        command.steering = self.prediction[0, 0]
        command.accelerator = self.prediction[0, 1]

        command.gear = self.caja_cambios(carstate.rpm, carstate.gear, carstate.current_lap_time)

        if carstate.distance_from_center < -1 or 1 < carstate.distance_from_center or \
                (carstate.current_lap_time > 3 and carstate.speed_x < 1):
            command.meta = 1

        return command

    @staticmethod
    def caja_cambios(rpm, gear, time) -> int:
        # Controlar
        if time < 2:
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

        # Función fitness
        fit = genetic.fitness_func(self.distance_raced, speed_av, self.lap, raycast_av)
        print(fit)
        genetic.guardar_fitness(fit)

        print('-------------------------------')
