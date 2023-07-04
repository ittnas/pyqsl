import pytest
import pyqsl


def test_simulation_result_creation(dataset):
    result = pyqsl.SimulationResult(dataset)
    assert result.foo.shape == dataset.foo.shape
    assert 'foo' in dir(result)


def test_attribute_error_is_risen(dataset):
    result = pyqsl.SimulationResult(dataset)
    with pytest.raises(AttributeError):
        result.xyz


def test_get_settings(simulation_result):
    simulation_result.settings.a


def test_setting_broadcasting(simulation_result):
    assert simulation_result.d.shape == simulation_result.dataset['diff'].shape # Sweep
    assert simulation_result.b.shape == simulation_result.dataset['diff'].shape
    assert simulation_result.c.shape == simulation_result.dataset['diff'].shape # Sweep
    assert simulation_result.h.shape == simulation_result.dataset['diff'].shape # Unrelated
