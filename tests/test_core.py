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


def test_run():
    settings = pyqsl.Settings()
    settings.a = 3
    settings.b = 2
    result = pyqsl.run(simple_task, settings)
    assert result == 5


def test_sweeps(ab_settings):
    sweeps = {ab_settings.a.name: np.linspace(0, 1, 3),
              ab_settings.b: np.linspace(-1, 0, 5)
              }
    result = pyqsl.run(simple_task, ab_settings, sweeps=sweeps, expand_data=False)
    assert result.shape == (3, 5)
    print(result)
