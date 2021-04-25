from pytocl.driver import Driver
from pytocl.car import State, Command
import numpy as np


class MyDriver(Driver):

    def drive(self, carstate: State) -> Command:
        command = Command()

        command.accelerator = 0.9
        command.gear = 1
        command.brake = 0.1

        raycasts = np.array(carstate.focused_distances_from_edge)

        if np.average(raycasts) != -1:
            print(raycasts)
        '''
        #self.steer(carstate, 0.0, command)

        # ACC_LATERAL_MAX = 6400 * 5
        # v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        v_x = 80

        self.accelerate(carstate, v_x, command)
        '''

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command
