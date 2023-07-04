import pytest
import pyqsl
import os


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


def test_get_sweeps(simulation_result):
    assert set(simulation_result.sweeps.keys()) == {'c', 'd'}


def test_setting_broadcasting(simulation_result):
    assert simulation_result.d.shape == simulation_result.dataset['diff'].shape # Sweep
    assert simulation_result.b.shape == simulation_result.dataset['diff'].shape
    assert simulation_result.c.shape == simulation_result.dataset['diff'].shape # Sweep
    assert simulation_result.h.shape == simulation_result.dataset['diff'].shape # Unrelated


def test_saving(simulation_result, tmpdir):
    simulation_result.save(os.path.join(tmpdir, 'save_test.pickle'))


def test_loading(simulation_result, tmpdir):
    path = os.path.join(tmpdir, 'save_test.pickle')
    simulation_result.save(path)
    loaded_result = pyqsl.load(path)
    loaded_result.settings.a
    assert dir(loaded_result) == ['a_evaluated', 'b_evaluated', 'c_evaluated', 'dataset', 'diff', 'e_evaluated', 'settings', 'sweeps']
