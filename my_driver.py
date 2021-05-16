from pytocl.driver import Driver
from pytocl.protocol import Client
from pytocl.car import State, Command
from modelo import Modelo
import numpy as np


class MyDriver(Driver):

    def drive(self, carstate: State) -> Command:
        self.carState = carstate
        command = Command()

        command.accelerator = 1
        command.gear = 1
        command.brake = 0
        # command.meta = 1

        raycasts = np.array(carstate.focused_distances_from_edge)

        if np.average(raycasts) != -1:
            Modelo.crear_modelo(self, raycasts, carstate.speed_x)



        return command

    def on_restart(self):
        ...
