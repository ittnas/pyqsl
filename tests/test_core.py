from typing import Any

import numpy as np
from pluggy import _result
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


def test_task_that_returns_settings(ab_settings):
    sweeps = {
        ab_settings.a.name: np.linspace(0, 1, 3),
        ab_settings.b: np.linspace(-1, 0, 5),
    }

    def task(a, b):
        settings = pyqsl.Settings()
        settings.c = a + b
        return settings

    result = pyqsl.run(task, ab_settings, sweeps=sweeps)
    assert result.c.shape == (3, 5)


def test_task_that_uses_settings(ab_settings):
    sweeps = {
        ab_settings.a.name: np.linspace(0, 1, 3),
        ab_settings.b: np.linspace(-1, 0, 5),
    }

    def task(settings):
        settings.c = settings.a + settings.b
        return settings

    result = pyqsl.run(task, ab_settings, sweeps=sweeps)
    assert result.c.shape == (3, 5)


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


def test_run_with_complicated_shapes_in_return():
    def task(a):
        return {"b": [a, a, a]}

    settings = pyqsl.Settings()
    settings.a = 2
    result = pyqsl.run(task, settings)
    assert (result.b == [settings.a, settings.a, settings.a]).all()

    def task(a):
        return {"b": [[a, a], a, a]}

    settings = pyqsl.Settings()
    settings.a = 2
    result = pyqsl.run(task, settings)
    assert result.b[0] == [settings.a, settings.a]
    sweeps = {"a": [0, 1]}
    result = pyqsl.run(task, settings, sweeps=sweeps)
    assert result.b[0][0] == [sweeps["a"][0], sweeps["a"][0]]

    def task(a):
        return [[a, a], a, a]

    settings = pyqsl.Settings()
    settings.a = 2
    result = pyqsl.run(task, settings)
    assert result.data[0] == [settings.a, settings.a]
    sweeps = {"a": [0, 1]}
    result = pyqsl.run(task, settings, sweeps=sweeps)
    assert result.data[0][0] == [sweeps["a"][0], sweeps["a"][0]]


def test_settings_with_reserved_names():
    def task(data):
        return data + 1

    settings = pyqsl.Settings()
    settings.data = 0
    result = pyqsl.run(task, settings)
    assert result.data == settings.data + 1
    assert result.settings.data == settings.data

    def task(a):
        return {"copy": a + 1}

    settings = pyqsl.Settings()
    settings.a = 0
    result = pyqsl.run(task, settings)
    assert result.copy == settings.a + 1

    def task(copy):
        return {"a": copy + 1}

    settings = pyqsl.Settings()
    settings.copy = 0
    with pytest.raises(TypeError):
        result = pyqsl.run(task, settings)


def test_dimensions_with_relations():
    settings = pyqsl.Settings()
    settings.a = [0, 1]
    settings.y = pyqsl.Setting(value=2, dimensions=["a"])
    assert settings.y.dimensions == ["a"]
    settings = pyqsl.Settings()
    settings.a = [0, 1]
    settings.y = 2
    settings.y.dimensions = ["a"]
    assert settings.y.dimensions == ["a"]
    settings = pyqsl.Settings()
    settings.a = [0, 1]
    settings.b = pyqsl.Setting(relation="a", dimensions=["a"])
    settings.c = 0
    result = pyqsl.run(None, settings=settings)
    assert result.dataset.b.shape == (2,)
    result = pyqsl.run(None, settings=settings, sweeps={"c": [0, 1, 2]})
    assert result.dataset.b.shape == (2,)
    assert result.dataset.c.shape == (3,)


def test_that_setting_shape_is_correct():
    settings = pyqsl.Settings()
    settings.a = 2
    settings.b = 3
    settings.c = pyqsl.Setting(relation="a + b")
    result = pyqsl.run(None, settings, sweeps=dict(a=np.linspace(0, 1, 3), b=[0, 2]))
    assert result.c.shape == (3, 2)
    result.dataset.c.plot()


def test_that_setting_shape_is_correct_for_complicated_shape():
    def task(settings):
        return {'g': settings.b.value}
    settings = pyqsl.Settings()
    settings.a = 2
    settings.b = 3
    settings.c = pyqsl.Setting(relation="a + b")
    settings.d = pyqsl.Setting(relation="a + b + c")
    settings.e = 3
    settings.f = pyqsl.Setting(relation='e + b')
    result = pyqsl.run(task, settings, sweeps=dict(a=np.linspace(0, 1, 3), b=[0, 2]))
    assert result.d.shape == (3, 2)
    result.dataset.d.plot()
    assert result.f.shape == (2, )
    result.dataset.f.plot()
    assert result.dataset.d.shape == (3, 2)


def test_sweeping_setting_with_dimension():
    settings = pyqsl.Settings()
    settings.a = [0, 1, 2]
    settings.b = pyqsl.Setting(dimensions=['a'], value=[0, 1, 2])
    settings.c = pyqsl.Setting(dimensions=['a'], relation='b')
    result = pyqsl.run(None, settings=settings, sweeps={'b': [[0, 1, 2], [0.5, 1.5, 2.5]]})
    assert result.c.shape == (2, 3)


def test_sweeping_with_mixed_type():
    def func(a):
        return a
    settings = pyqsl.Settings()
    settings.a = 2
    settings.b = pyqsl.Setting(relation=pyqsl.Function(function=func))
    result = pyqsl.run(None, settings, sweeps = {'a': [[0, 1], {0, 2}]})
    assert result.b[0] == [0, 1]
    assert result.b[1] == {0, 2}
    assert result.dataset.b.dims == ('a', )

def test_list_of_tasks_mode():
    def task_c(a):
        return {'c': a}
    def task_d(b):
        return {'d': b}
    settings = pyqsl.Settings()
    settings.a = 2
    settings.b = 3
    result = pyqsl.run(task_c, settings)
    assert result.c == settings.a
    result = pyqsl.run(None, settings)
    result = pyqsl.run([task_c, task_d], settings)
    assert result.c == settings.a
    assert result.d == settings.b
