import logging

from pytocl.analysis import DataLogWriter
import numpy as np
from modelo import Modelo

_logger = logging.getLogger(__name__)


class Driver:
    """
    Driving logic.
    Implement the driving intelligence in this class by processing the current
    car state as inputs creating car control commands as a response. The
    ``drive`` function is called periodically every 20ms and must return a
    command within 10ms wall time.
    """

    def __init__(self, logdata=True, generation=-1, n=-1, max_gear=1):
        self.generation = generation
        self.n = n
        self.max_gear = max_gear
        self.lap = 1
        self.speed_av = np.zeros(15)
        self.raycast_av = np.zeros(15)
        self.equidistance_av = np.zeros(2)
        self.angle_av = np.zeros(2)
        self.distance_raced = 0.0
        self.time = 0.0
        self.time_pred = 1.0
        self.prediction = np.zeros((1, 3))
        self.modelo = Modelo(generation, n)
        self.data_logger = DataLogWriter() if logdata else None

    @property
    def range_finder_angles(self):
        """Iterable of 19 fixed range finder directions [deg].
        The values are used once at startup of the client to set the directions
        of range finders. During regular execution, a 19-valued vector of track
        distances in these directions is returned in ``state.State.tracks``.
        """
        return -90, -45, 0, 45, 90

    def on_shutdown(self):
        """Server requested driver shutdown.
        Optionally implement this event handler to clean up or write data
        before the application is stopped.
        """
        if self.data_logger:
            self.data_logger.close()
            self.data_logger = None
