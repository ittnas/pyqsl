from typing import Any

import numpy as np
import pytest

import pyqsl


@pytest.fixture()
def ab_settings():
    settings = pyqsl.Settings()
    settings.a = 2
    settings.b = 3
    return settings


def simple_task(a: int, b: int) -> int:
    return a + b


def more_complicated_task(a: Any, b: Any, c: Any) -> Any:
    return (a + b) * c


def task_that_returns_tuples(a: float, b: float) -> tuple[float, float]:
    return a + b, a - b


def task_that_returns_dicts(a: float, b: float) -> dict[str, float]:
    return {"sum": a + b, "diff": a - b}


def add_new_setting(settings: pyqsl.Settings):
    settings.c = 5.0


def adjust_settings(settings: pyqsl.Settings):
    settings.c = 3.0


def task_with_settings_as_input(settings: pyqsl.Settings):
    settings_as_namespace = settings.to_namespace()
    return settings_as_namespace.a * settings_as_namespace.b


def update_output(output, settings):
    return {"result": output}


def test_run():
    settings = pyqsl.Settings()
    settings.a = 3
    settings.b = 2
    simulation_result = pyqsl.run(simple_task, settings, expand_data=False)
    assert simulation_result.data == 5
    assert np.isscalar(simulation_result.data)
    simulation_result = pyqsl.run(simple_task, settings, expand_data=True)
    assert simulation_result.data == 5
    assert np.isscalar(simulation_result.data)


def test_sweeps(ab_settings):
    sweeps = {
        ab_settings.a.name: np.linspace(0, 1, 3),
        ab_settings.b: np.linspace(-1, 0, 5),
    }
    result = pyqsl.run(simple_task, ab_settings, sweeps=sweeps, expand_data=False)
    assert result.data.shape == (3, 5)
    result = pyqsl.run(simple_task, ab_settings, sweeps=sweeps, expand_data=True)
    assert result.data.shape == (3, 5)


def test_parallel_execution(ab_settings):
    sweeps = {
        ab_settings.a.name: np.linspace(0, 1, 3),
    }
    result = pyqsl.run(
        simple_task, ab_settings, sweeps=sweeps, expand_data=False, parallelize=True
    )
    assert result.data.shape == (3,)
    assert np.all(result.data == [3, 3.5, 4.0])


def test_parallel_n_cores(ab_settings):
    sweeps = {
        ab_settings.a.name: np.linspace(0, 1, 3),
    }
    result = pyqsl.run(
        simple_task,
        ab_settings,
        sweeps=sweeps,
        expand_data=False,
        parallelize=True,
        n_cores=-500,
    )
    result = pyqsl.run(
        simple_task,
        ab_settings,
        sweeps=sweeps,
        expand_data=False,
        parallelize=True,
        n_cores=4,
    )
    result = pyqsl.run(
        simple_task,
        ab_settings,
        sweeps=sweeps,
        expand_data=False,
        parallelize=True,
        n_cores=-1,
    )
    with pytest.raises(ValueError):
        result = pyqsl.run(
            simple_task,
            ab_settings,
            sweeps=sweeps,
            expand_data=False,
            parallelize=True,
            n_cores=0,
        )


def test_task_that_returns_tuples(ab_settings):
    sweeps = {
        ab_settings.a.name: np.linspace(0, 1, 3),
        ab_settings.b: np.linspace(-1, 0, 5),
    }
    result = pyqsl.run(
        task_that_returns_tuples, ab_settings, sweeps=sweeps, expand_data=True
    )
    assert result.data[0].shape == (3, 5)  # First element of tuple
    assert result.data[1].shape == (3, 5)  # Second element of tuple
    result = pyqsl.run(
        task_that_returns_tuples, ab_settings, sweeps=sweeps, expand_data=False
    )
    assert result.data.shape == (3, 5)
    assert result.data[0, 0] == (-1, +1)


def test_task_that_returns_dicts(ab_settings):
    sweeps = {
        ab_settings.a.name: np.linspace(0, 1, 3),
        ab_settings.b: np.linspace(-1, 0, 5),
    }
    result = pyqsl.run(
        task_that_returns_dicts, ab_settings, sweeps=sweeps, expand_data=True
    )
    assert result.sum.shape == (3, 5)  # First element of tuple
    assert result.diff.shape == (3, 5)  # Second element of tuple
    result = pyqsl.run(
        task_that_returns_dicts, ab_settings, sweeps=sweeps, expand_data=False
    )
    assert result.data.shape == (3, 5)
    assert result.data[0, 0]["sum"] == -1.0


def test_prepocessing(ab_settings):
    result = pyqsl.run(
        more_complicated_task, ab_settings, pre_process_before_loop=add_new_setting
    )
    # Chech that c is not added to settings
    with pytest.raises(AttributeError):
        ab_settings.c

    assert result.data == 25


def test_prepocess_in_loop(ab_settings):
    sweeps = {
        ab_settings.a.name: np.linspace(0, 1, 3),
    }
    result = pyqsl.run(
        more_complicated_task,
        ab_settings,
        sweeps=sweeps,
        pre_process_in_loop=adjust_settings,
    )
    assert (result.data == [9, 10.5, 12]).all()


def test_post_process_in_loop(ab_settings):
    sweeps = {
        ab_settings.a.name: np.linspace(0, 1, 3),
    }
    result = pyqsl.run(
        more_complicated_task,
        ab_settings,
        sweeps=sweeps,
        pre_process_in_loop=adjust_settings,
        post_process_in_loop=update_output,
    )
    assert result.result.shape == (3,)


def test_simulation_with_relations(ab_settings):
    sweeps = {ab_settings.a: np.linspace(0, 1, 3)}
    ab_settings.b.relation = pyqsl.Equation(equation="a + 1")
    result = pyqsl.run(simple_task, ab_settings, sweeps=sweeps)
    assert (result.data == [1, 2, 3]).all()


def test_get_invalid_args():
    args = {"a": 1, "b": 2, "c": 3}

    def task1(a, b):
        pass

    def task2(a, b, **kw):
        return kw["c"]

    def task3(a, b, c=2):
        return c

    def task4(a=1):
        pass

    assert pyqsl.core._get_invalid_args(task1, args) == set("c")
    assert pyqsl.core._get_invalid_args(task2, args) == set()
    assert pyqsl.core._get_invalid_args(task3, args) == set()
    assert pyqsl.core._get_invalid_args(task4, args) == {"c", "b"}


def test_settings_as_task_argument(ab_settings):
    result = pyqsl.run(task_with_settings_as_input, ab_settings)
    assert result.data == 6


def test_add_dimensions_and_run():
    def task(x, a):
        return {"y": a * x}

    settings = pyqsl.Settings()
    settings.x = np.linspace(0, 1, 7)
    settings.x.unit = "m"
    settings.y = None
    settings.y.dimensions = ["x"]
    settings.y.unit = "m"
    settings.a = 2
    sweeps = {"a": [1, 2]}
    result = pyqsl.run(task, settings, sweeps=sweeps)
    assert result.dataset.y.dims == ("a", "x")
