import pytest
import pyqsl
import xarray as xr
import numpy as np


@pytest.fixture
def task():
    def task_to_return(a, b):
        return {'diff': a - b}
    return task_to_return


@pytest.fixture
def settings():
    settings = pyqsl.Settings()
    settings.amplitude = 4
    settings.amplitude.unit = 'V'
    settings.frequency = 2
    settings.frequency.unit = 'Hz'
    return settings


@pytest.fixture
def settings_with_relations():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.b = 2
    settings.c = 3
    settings.d = 4
    settings.e = 5
    settings.h = 6
    settings.a.relation = 'b + c'
    settings.b.relation = 'c'
    settings.e.relation = 'a'
    return settings


@pytest.fixture
def dataset():
    data = xr.DataArray(np.ones((2, 3)), coords={'x': ['a', 'b']}, dims=('x', 'y'))
    ds = xr.Dataset({'foo': data, 'bar': ('x', [1, 2]), 'baz': np.pi})
    return ds


@pytest.fixture
def simulation_result(task, settings_with_relations):
    sweeps = {'c': np.linspace(0, 1, 7), 'd': np.linspace(-7, 0, 11)}
    result = pyqsl.run(task, settings=settings_with_relations, sweeps=sweeps)
    return result

