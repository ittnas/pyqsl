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


class SimulationResult():
    """
    This class stores the simulation results and the settings used to generate it.
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
        if key == 'settings':
            return self.dataset.attrs['settings']
        if key in self.dataset.data_vars:
            da = self.dataset[key]
            if len(da.dims) == 0:
                return da.values[()]
            return da.values
        if 'settings' in self.dataset.attrs and key in self.dataset.attrs['settings']:
            if f'{key}_evaluated' in self.dataset.data_vars:
                return self.dataset[f'{key}_evaluated'].values if len(self.dataset[f'{key}_evaluated'].dims) > 0 else self.dataset[f'{key}_evaluated'].values[()]
            model_data_array = self.dataset[list(self.dataset.data_vars)[0]]
            if key in self.dataset.coords:
                return np.broadcast_to(np.array(self.dataset.coords[key].values), model_data_array.shape, subok=True)
            return np.broadcast_to(self.dataset.attrs['settings'][key].value, model_data_array.shape, subok=True)
        raise AttributeError()

    def __dir__(self):
        """
        Add attributes created by __getattr__.
        """
        items = list(self.__dict__.keys()) + list(self.dataset.data_vars)
        if 'settings' in self.dataset.attrs:
            items = items + ['settings']
        return items

    def __str__(self):
        return str(self.dataset)
