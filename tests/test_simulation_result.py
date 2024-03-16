import os

import numpy as np
import pytest

import pyqsl


def test_simulation_result_creation(dataset):
    result = pyqsl.SimulationResult(dataset)
    assert result.foo.shape == dataset.foo.shape
    assert "foo" in dir(result)


def test_attribute_error_is_risen(dataset):
    result = pyqsl.SimulationResult(dataset)
    with pytest.raises(AttributeError):
        result.xyz


def test_get_settings(simulation_result):
    simulation_result.settings.a


def test_get_sweeps(simulation_result):
    assert set(simulation_result.sweeps.keys()) == {"c", "d"}


def test_setting_broadcasting(simulation_result):
    assert simulation_result.d.shape == (11,)  # Sweep
    assert simulation_result.b.shape == (7,)
    assert simulation_result.c.shape == (7,)  # Sweep
    assert np.isscalar(simulation_result.h)  # Unrelated


def test_saving(simulation_result, tmpdir):
    simulation_result.save(os.path.join(tmpdir, "save_test.pickle"))


def test_loading(simulation_result, tmpdir):
    path = os.path.join(tmpdir, "save_test.pickle")
    simulation_result.save(path)
    loaded_result = pyqsl.load(path)
    loaded_result.settings.a
    assert set(dir(loaded_result)) == set(
        [
            "a",
            "b",
            "c",
            "d",
            "e",
            "h",
            "dataset",
            "diff",
            "settings",
            "sweeps",
        ]
    )


def test_get(simulation_result):
    a, diff = simulation_result.get(["a", "diff"])
    assert a.shape == (7, 11)
    assert diff.shape == (7, 11)

    a, diff = simulation_result.get(["a", "diff"], order=["d", "c"])
    assert a.shape == (11, 7)
    assert diff.shape == (11, 7)

    a, diff = simulation_result.get(["a", "diff"], order=["d", "c", "a"])
    assert a.shape == (11, 7, 1)
    assert diff.shape == (11, 7, 1)

    a, diff = simulation_result.get(["a", "diff"], order=["d", "a", "c"])
    assert a.shape == (11, 1, 7)
    assert diff.shape == (11, 1, 7)

    assert simulation_result.get("diff").shape == (7, 11)


def test_to_settings():
    settings = pyqsl.Settings()
    settings.a = [0, 1, 2]
    settings.b = None
    settings.b.relation = "a"
    settings.b.dimensions = ["a"]
    settings.c = 2

    def task(a, b, c):
        settings = pyqsl.Settings()
        settings.d = np.mean(a) + np.mean(b) + c
        return settings

    result = pyqsl.run(task, settings=settings, sweeps={"c": [0, 1, 2]})
    settings = result.to_settings()
    assert settings.d.dimensions == ["c"]
    assert settings.b.dimensions == ["a"]

    def new_task(d):
        settings = pyqsl.Settings()
        settings.e = d
        return settings

    settings.e = None
    settings.e.dimensions = ["c"]
    new_settings = pyqsl.run(new_task, settings=settings).to_settings()
    assert new_settings.e.dimensions == ["c"]
