from pytocl.driver import Driver
from pytocl.car import State, Command
import numpy as np


class MyDriver(Driver):

    def drive(self, carstate: State) -> Command:
        command = Command()

        raycasts = np.array(carstate.focused_distances_from_edge)

        if np.average(raycasts) != -1:
            self.prediction = self.modelo.inferir_modelo(raycasts, carstate.speed_x)
            '''
            # print(f'Prediction:{self.prediction}')
            # print(f'Speed:{carstate.speed_x}')
            '''

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

        print('-------------------------------')
