import logging

import math

from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
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

    def __init__(self, logdata=True, generation=-1, n=-1):
        self.generation = generation
        self.n = n
        self.lap = 1
        self.speed_av = np.zeros(2)
        self.raycast_av = np.zeros(2)
        self.distance_raced = 0.0
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
        return -90, -75, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, \
            30, 45, 60, 75, 90

    def on_shutdown(self):
        """Server requested driver shutdown.
        Optionally implement this event handler to clean up or write data
        before the application is stopped.
        """
        if self.data_logger:
            self.data_logger.close()
            self.data_logger = None
