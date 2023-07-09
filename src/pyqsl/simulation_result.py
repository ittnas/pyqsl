"""
Defines a class to store simulation results.

Classes:

    SimulationResult
"""
import logging
import pickle
from typing import Any, Optional, Union

import numpy as np
import xarray as xr

from pyqsl.settings import Setting

logger = logging.getLogger(__name__)


class SimulationResult:
    """
    This class stores the simulation results and the settings used to generate it.

    Simulation result is a wrapper around xarray, which is used as the main data structure.
    The underlying xarray dataset can be accessed through ```dataset``` attribute.

    The settings, results and sweeps can be accessed as attributes of the simulation result.

    There are two ways of accesssing settings, one can either directly call e.g. ```simulation_result.amplitude```
    in which case the only setting values are returned and broadcasted to sweep dimensions.

    Another option is to access ```simulation_result.settings.amplitude```, which returns the setting object.
    """

    dataset: xr.Dataset

    def __init__(self, dataset: Optional[xr.Dataset] = None):
        if dataset is None:
            self.dataset = xr.Dataset()
        else:
            self.dataset = dataset

    def __getattr__(self, key: str) -> Any:
        """
        Returns attributes of the simulation result.

        If key is a data_var fetches the result from the underlying Dataset.
        If key is a setting, looks for "settings" under .attrs attribute of the dataset and returns
        the setting value. The dimensions of the setting are expanded to correspond to the sweep dimensions.
        If looking for a setting that is not a part of a relation, all the returned values in the array will
        have the same value. If setting is either part of a sweep or a relation, the values of the returned array
        will correspond to the setting value at each point of the n-dimensional sweep.

        Args:
            key: key to fetch.
        """
        # pylint: disable=too-many-return-statements
        if key == "settings":
            try:
                return self.dataset.attrs["settings"]
            except KeyError as exc:
                raise AttributeError(
                    "Settings not found in simulation result."
                ) from exc
        if key == "sweeps":
            return {coord: value.values for coord, value in self.dataset.coords.items()}
        if key in self.dataset.data_vars:
            da = self.dataset[key]
            if len(da.dims) == 0:
                return da.values[()]
            return da.values
        if "settings" in self.dataset.attrs and key in self.dataset.attrs["settings"]:
            if f"{key}_evaluated" in self.dataset.data_vars:
                return (
                    self.dataset[f"{key}_evaluated"].values
                    if len(self.dataset[f"{key}_evaluated"].dims) > 0
                    else self.dataset[f"{key}_evaluated"].values[()]
                )
            model_data_array = self.dataset[list(self.dataset.data_vars)[0]]
            if key in self.dataset.coords:
                broadcast_shape = [1] * len(model_data_array.shape)
                broadcast_shape[model_data_array.dims.index(key)] = -1
                reshaped = np.reshape(
                    np.array(self.dataset.coords[key].values), broadcast_shape
                )
                return np.broadcast_to(
                    reshaped,
                    model_data_array.shape,
                    subok=True,
                )
            return np.broadcast_to(
                self.dataset.attrs["settings"][key].value,
                model_data_array.shape,
                subok=True,
            )
        raise AttributeError()

    def __dir__(self):
        """
        Add attributes created by __getattr__.
        """
        setting_names = (
            [setting.name for setting in self.settings]
            if hasattr(self, "settings")
            else []
        )
        items = (
            list(self.__dict__.keys()) + list(self.dataset.data_vars) + setting_names
        )
        if "settings" in self.dataset.attrs:
            items = items + ["settings"] + ["sweeps"]
        return items

    def __str__(self):
        return str(self.dataset)

    def save(self, path: str):
        """
        Saves the simulation result to a pickle file.

        Args:
            path: savepath
        """
        with open(path, "wb") as f:
            pickle.dump(self.dataset, f)

    def get(
        self,
        key: Union[str, tuple[str]],
        order: Optional[tuple[Union[str, Setting]]] = None,
    ):
        """
        Returns simulation parameters reordered according given order.

        Args:
            key: Attribute to fetch or a tuple of attributes to fetch.
            order:
                List of settings or setting names according to which the returned data is ordered.
                If None, the default order is used. If the setting name is not part of the sweep,
                the dimensions of the returned data will be expanded so that returned data
                will always have at least as many dimensions as there are settings listed here.

        Returns:
            Returned data. The return value will have the same dimensions as the key argument, i.e.
            for a string input returns data correspond to the key, and for a tuple input returns a
            tuple of data arrays.
        """
        if isinstance(key, str):
            return self._reorder(key, order)
        data = []
        for key_element in key:
            data.append(self._reorder(key_element, order))
        return tuple(data)

    def _reorder(self, key, order: Optional[tuple[Union[str, Setting]]] = None):
        if not hasattr(self, key):
            raise ValueError(f"Key ({key}) must be a setting or data variable.")
        if order is None:
            return getattr(self, key)
        order_names = [
            order_name if isinstance(order_name, str) else order_name.name
            for order_name in order
        ]
        for order_name in order_names:
            if order_name not in self.settings:
                raise ValueError("All elements of order argument must be in settings")
        current_dimension_names = list(self.sweeps.keys())
        new_dimension_names = [
            order_name
            for order_name in order_names
            if order_name not in current_dimension_names
        ]
        all_dimension_names_in_old_order = current_dimension_names + new_dimension_names
        slice_tuple = tuple(
            slice(None) if dim in current_dimension_names else np.newaxis
            for dim in all_dimension_names_in_old_order
        )
        data = getattr(self, key)
        data = data[slice_tuple]
        transpose_list = []
        for order_name in order_names:
            transpose_list.append(all_dimension_names_in_old_order.index(order_name))
        other_dimensions = [
            element
            for element in list(range(len(all_dimension_names_in_old_order)))
            if element not in transpose_list
        ]
        transpose_list.extend(other_dimensions)
        data = np.transpose(data, tuple(transpose_list))
        return data


def load(path: str) -> SimulationResult:
    """
    Loads the dataset from pickle file.

    Args:
        path: Load path
    """
    with open(path, "rb") as f:
        dataset = pickle.load(f)
    simulation_result = SimulationResult(dataset)
    return simulation_result
