"""
Defines a class to store simulation results.
Classes:
    SimulationResult
"""
from dataclasses import dataclass
from typing import Any, Optional
import xarray as xr
from pyqsl.settings import Settings
import numpy as np
import copy
import dataclasses
import json
import pickle


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

    def __getattr__(self, key):
        """
        Getting attribute of the simulation result fetches a data_var from the underlying Dataset.
        """
        if key == "settings":
            return self.dataset.attrs["settings"]
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
                broadcast_shape = [1]*len(model_data_array.shape)
                broadcast_shape[model_data_array.dims.index(key)] = -1
                reshaped = np.reshape(np.array(self.dataset.coords[key].values), broadcast_shape)
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
        items = list(self.__dict__.keys()) + list(self.dataset.data_vars)
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
