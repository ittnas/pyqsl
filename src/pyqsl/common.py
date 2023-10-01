"""
Definitions used by more than one other module.
"""
import logging
from typing import Any, Sequence

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

    Args:
        sweeps: Sweeps dictionary which keys can either be setting or setting name.

    Returns:
        A soft copy of sweeps in new format.
    """
    return {
        key.name if isinstance(key, Setting) else key: value
        for key, value in sweeps.items()
    }


def convert_data_coordinates_to_standard_form(
    data_coordinates: DataCoordinatesType | DataCoordinatesStandardType,
) -> DataCoordinatesStandardType:
    """
    Converts data coordinates from verbose type to standard type

    Args:
        data_coordinates: Mapping from data coordinates to their additional sweeps.

    Returns:
        A soft copy of data_coordinates in new format.
    """
    new_data_coordinates = {}
    for key, value in data_coordinates.items():
        new_key = key.name if isinstance(key, Setting) else key
        new_value = convert_sweeps_to_standard_form(value)
        new_data_coordinates[new_key] = new_value
    return new_data_coordinates
