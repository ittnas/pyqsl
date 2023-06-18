import pytest
import pyqsl
import numpy as np

@pytest.fixture()
def ab_settings():
    settings = pyqsl.Settings()
    settings.a = 2
    settings.b = 3
    return settings


def simple_task(a, b):
    return a + b


def task_that_returns_tuples(a, b):
    return a+b, a-b


def task_that_returns_dicts(a, b):
    return {'sum': a+b, 'diff': a - b}


def test_run():
    settings = pyqsl.Settings()
    settings.a = 3
    settings.b = 2
    result = pyqsl.run(simple_task, settings, expand_data=False)
    assert result == 5
    assert isinstance(result, int)
    result = pyqsl.run(simple_task, settings, expand_data=True)
    assert result == 5
    assert isinstance(result, int)


def test_sweeps(ab_settings):
    sweeps = {ab_settings.a.name: np.linspace(0, 1, 3),
              ab_settings.b: np.linspace(-1, 0, 5)
              }
    result = pyqsl.run(simple_task, ab_settings, sweeps=sweeps, expand_data=False)
    assert result.shape == (3, 5)
    result = pyqsl.run(simple_task, ab_settings, sweeps=sweeps, expand_data=True)
    assert result.shape == (3, 5)


def test_parallel_execution(ab_settings):
    sweeps = {ab_settings.a.name: np.linspace(0, 1, 3),
              }
    result = pyqsl.run(simple_task, ab_settings, sweeps=sweeps, expand_data=False, parallelize = True)
    assert result.shape == (3,)
    assert np.all(result == [3, 3.5, 4.0])

def test_parallel_n_cores(ab_settings):
    sweeps = {ab_settings.a.name: np.linspace(0, 1, 3),
              }
    result = pyqsl.run(simple_task, ab_settings, sweeps=sweeps, expand_data=False, parallelize = True, n_cores=-500)
    result = pyqsl.run(simple_task, ab_settings, sweeps=sweeps, expand_data=False, parallelize = True, n_cores=4)
    result = pyqsl.run(simple_task, ab_settings, sweeps=sweeps, expand_data=False, parallelize = True, n_cores=-1)
    with pytest.raises(ValueError):
        result = pyqsl.run(simple_task, ab_settings, sweeps=sweeps, expand_data=False, parallelize = True, n_cores=0)


def test_task_that_returns_tuples(ab_settings):
    sweeps = {ab_settings.a.name: np.linspace(0, 1, 3),
              ab_settings.b: np.linspace(-1, 0, 5)
              }
    result = pyqsl.run(task_that_returns_tuples, ab_settings, sweeps=sweeps, expand_data=True)
    assert result[0].shape == (3, 5) # First element of tuple
    assert result[1].shape == (3, 5) # Second element of tuple
    result = pyqsl.run(task_that_returns_tuples, ab_settings, sweeps=sweeps, expand_data=False)
    assert result.shape == (3, 5, 2)


def test_task_that_returns_dicts(ab_settings):
    sweeps = {ab_settings.a.name: np.linspace(0, 1, 3),
              ab_settings.b: np.linspace(-1, 0, 5)
              }
    result = pyqsl.run(task_that_returns_dicts, ab_settings, sweeps=sweeps, expand_data=True)
    assert result['sum'].shape == (3, 5)  # First element of tuple
    assert result['diff'].shape == (3, 5)  # Second element of tuple
    result = pyqsl.run(task_that_returns_dicts, ab_settings, sweeps=sweeps, expand_data=False)
    assert result.shape == (3, 5)
    assert result[0, 0]['sum'] == -1.0
