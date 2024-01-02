"""
Package for running simulations and saving the results.

Usage
-----
The simulation task is performed in a user-defined funtion that takes a
dictionary of the simulation parameters as an input and gives simulation
result as an output.

The power of pyqsl is to enable the user to change the values of simulation
parameters in a loop. This is done in the run() function. The
run function takes a simulation task as an argument and loops it over with
all the different parameters defined by the user in the sweeps dictionary.
The keys of the dictionary are the parameters that are varied
(they need to exist also in the params dictionary) and the keys are lists of
parameter values.

The output of the run function is a SimulationResult instance which attributes
correspond to different simulation result arrays. The dimensions of the result
arreys correspond to different parameter configurations. There is also an option
to perform the simulation in parallel.

The results can be saved to a file using the ``save`` method in simulation result.
"""
import logging
from logging import NullHandler

import pint_xarray

from .core import run  # pylint: disable=cyclic-import
from .relation import Equation, LookupTable, Relation  # pylint: disable=cyclic-import
from .settings import Setting, Settings  # pylint: disable=cyclic-import
from .simulation_result import SimulationResult, load

if not pint_xarray.unit_registry.default_format:
    pint_xarray.unit_registry.default_format = "~"

logging.getLogger(__name__).addHandler(NullHandler())


def add_stderr_logger(level: int = logging.DEBUG) -> logging.StreamHandler:
    """
    Helper for quickly adding a StreamHandler to the logger. Useful for
    debugging. Returns the handler after adding it. Adapted from urllib3.
    """
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%d/%b/%Y %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.debug("Added a stderr logging handler to logger: %s", __name__)
    return handler


del NullHandler
