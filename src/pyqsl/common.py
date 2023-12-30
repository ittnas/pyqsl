"""
Definitions used by more than one other module.
"""
import logging
from typing import Any, Sequence
import numpy as np

from pyqsl.settings import Setting

SweepsType = dict[str | Setting, Sequence]
TaskOutputType = dict[str, Any] | tuple | list | Any
SweepsStandardType = dict[str, Sequence]
DataCoordinatesType = dict[Setting | str, SweepsType]
DataCoordinatesStandardType = dict[str, SweepsStandardType]
logger = logging.getLogger(__name__)


def convert_sweeps_to_standard_form(
    sweeps: SweepsType | SweepsStandardType,
) -> SweepsStandardType:
    """
    Converts sweeps from form that can contain settings to standard form that only has settings names.

    Additionally, ensures that conversion to numpy array does not add extra dimensions. This is required
    for using the sweep as a coordinate for xarray.

    Args:
        sweeps: Sweeps dictionary which keys can either be setting or setting name.

    Returns:
        A soft copy of sweeps in new format.

    Raises:
        ValueError if any name for a sweep is None.
    """
    names = [key.name if isinstance(key, Setting) else key for key in sweeps.keys()]
    if any(name is None for name in names):
        raise ValueError("Name for a sweep parameter must not be None.")

    # Tell mypy that there are no Nones
    names_without_none = [name for name in names if name is not None]
    new_sweep_values = []

    for sweep_values in sweeps.values():
        ii = 0
        for _ in sweep_values:
            ii += 1
        correct_shape = True
        try:
            array_size = np.array(sweep_values).size
            if array_size != ii:
                correct_shape = False
        except ValueError:
            correct_shape = False
        if not correct_shape:
            cast_shape = np.zeros((ii, ), dtype='O')
            for jj, value in enumerate(sweep_values):
                logger.debug(value)
                cast_shape[jj] = value
            new_sweep_values.append(cast_shape)
        else:
            new_sweep_values.append(sweep_values)
    return dict(zip(names_without_none, new_sweep_values))


def convert_data_coordinates_to_standard_form(
    data_coordinates: DataCoordinatesType | DataCoordinatesStandardType,
) -> DataCoordinatesStandardType:
    """
    Converts data coordinates from verbose type to standard type

    Args:
        data_coordinates: Mapping from data coordinates to their additional sweeps.

    Returns:
        A soft copy of data_coordinates in new format.

    Raises:
        ValueError if name for data coordinate is None.
    """
    new_data_coordinates = {}
    for key, value in data_coordinates.items():
        new_key = key.name if isinstance(key, Setting) else key
        if new_key is None:
            raise ValueError("Name for a data coordiante must not be None.")
        new_value = convert_sweeps_to_standard_form(value)
        new_data_coordinates[new_key] = new_value
    return new_data_coordinates
