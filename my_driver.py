from pytocl.driver import Driver
from pytocl.protocol import Client
from pytocl.car import State, Command
from modelo import Modelo
import numpy as np


class MyDriver(Driver):

    modelo = Modelo()

    def drive(self, carstate: State) -> Command:
        self.carState = carstate
        command = Command()

        raycasts = np.array(carstate.focused_distances_from_edge)

        if np.average(raycasts) != -1:
            self.prediction = self.modelo.inferir_modelo(raycasts, carstate.speed_x)
            print(self.prediction)

        command.steering = self.prediction[0, 0]
        command.accelerator = self.prediction[0, 1]
        command.brake = 0

        command.gear = 1
        # command.meta = 1

        return command

    def on_restart(self):
        ...
