"""
Package for running simulations and saving the results.

Usage
-----
The simulation task is performed in a user-defined funtion that takes a dictionary of the simulation parameters as an input and gives simulation result as an output.

The power of pyqsl is to enable the user to change the values of simulation parameters in a loop. This is done in the simulation_loop function. The simulation loop takes a simulation task as a parameter and loops it over with all the different parameters defined by the user in the sweep_arrays dictionary. The keys of the dictionary are the parameters that are varied (they need to exist also in the params dictionary) and the keys are lists of parameter values.

The output of the simulation loop is a list with length equal to the total number of different parameter configurations. There is also an option to perform the simulation in parallel.

The output list can be further saved by the saving functions provided. Some of the saving functions require the output list elements to have a specified structure, most notably the .hdf5 saving function that uses Labber API to save the data. The save_data_pickle function on the other hand saves the data in binary format, that can be loaded using the load_pickled_data function.

"""
from .core import run
from .settings import Setting, Settings
from .relation import Relation, Equation, LookupTable
from .simulation_result import SimulationResult

import logging
from logging import NullHandler
logging.getLogger(__name__).addHandler(NullHandler())


def add_stderr_logger(level: int = logging.DEBUG) -> logging.StreamHandler:
    """
    Helper for quickly adding a StreamHandler to the logger. Useful for
    debugging. Returns the handler after adding it. Adapted from urllib3.
    """
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s", datefmt="%d/%b/%Y %H:%M:%S",))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.debug("Added a stderr logging handler to logger: %s", __name__)
    return handler


del NullHandler
