"""
This is a small example that demonstrates different ways of defining a task and sweeps.
Instead of doing something sensible, this is more like a list of different possibilities
intended for inspiration.
"""
import numpy as np
import pyqsl

settings = pyqsl.Settings()
settings.a = 1
settings.b = 2
settings.e = 5
settings.f = 1
settings.h = [0, 1, 2]
settings.i = [4, 5, 6]
settings.i.dimensions = ["h"]
settings.k = np.zeros((3, 3))
settings.l = [6, 2, 3]
settings.k.dimensions = ["h", "l"]
settings.f.relation = pyqsl.Equation(equation="f")
settings.d = pyqsl.Setting()
settings.d.relation = pyqsl.Equation(equation="a+b")
settings.g = pyqsl.Setting()
settings.g.relation = pyqsl.Equation(equation="d+a")
settings.e = pyqsl.Setting()
settings.e.relation = pyqsl.Equation(equation="a")

#sweep_arrays = {"a": [0, 1, 2], "b": [0.5, 1.5]}
#sweep_arrays = {"a": [[0, 1], [0, 2], [0, 3]], "b": [0.5, 1.5]}
#sweep_arrays = {"a": [1,  2, 3], "b": [{"first", "second"}, {"third", "fourth"}]}
sweep_arrays = {}
##
def task(a, b):
    return [a, b, a]


output = pyqsl.run(
    task,
    settings=settings,
    sweeps=sweep_arrays,
    parallelize=False,
    expand_data=True, # If set to False, will not succeed.
)


##
def task(a, b):
    settings = pyqsl.Settings()
    settings.c = a + b
    return settings


output = pyqsl.run(
    task,
    settings=settings,
    sweeps=sweep_arrays,
    parallelize=False,
)


##
def task(a, b):
    return a


output = pyqsl.run(
    task,
    settings=settings,
    sweeps=sweep_arrays,
    parallelize=False,
    expand_data=False,
)


##
def task(a, b):
    return {"c": 4}


output = pyqsl.run(
    task,
    settings=settings,
    sweeps=sweep_arrays,
    parallelize=False,
    expand_data=False,
)


##
def task(a, b):
    return {"c": 4}


output = pyqsl.run(
    task,
    settings=settings,
    sweeps=sweep_arrays,
    parallelize=False,
    expand_data=True,
)


##
def task(a, b):
    return [a, b, a, b, a, b]


output = pyqsl.run(
    task,
    settings=settings,
    sweeps=sweep_arrays,
    parallelize=False,
    expand_data=True,
)


##
def task(a, b):
    return [[a, a], b, a, b, a, b]


output = pyqsl.run(
    task,
    settings=settings,
    sweeps=sweep_arrays,
    parallelize=False,
    expand_data=True,
)


##
def task(a, b):
    return a


output = pyqsl.run(
    task,
    settings=settings,
    sweeps=sweep_arrays,
    parallelize=False,
    expand_data=True,
)


##
def task(a, b):
    return {"c": [0, 1, 2, 3]}


output = pyqsl.run(
    task,
    settings=settings,
    sweeps=sweep_arrays,
    parallelize=False,
    expand_data=True,
)


##
def task(a, b):
    return {"c": [[0, 1], [1, 1], [2, 1], [3, 1]]}


output = pyqsl.run(
    task,
    settings=settings,
    sweeps=sweep_arrays,
    parallelize=False,
    expand_data=True,
)


##
def task(a, b):
    if a == 0:
        return {"c": [0, 0]}
    return {"c": [1, 1, 1]}


output = pyqsl.run(
    task,
    settings=settings,
    sweeps=sweep_arrays,
    parallelize=False,
    expand_data=True,
)


##
def task(a, b):
    return {"i": [1, 2, 3]}


output = pyqsl.run(
    task,
    settings=settings,
    sweeps=sweep_arrays,
    parallelize=False,
    expand_data=True,
)


##
def task(a, b):
    return {"diff": a - b}


settings = pyqsl.Settings()
settings.a = 1
settings.b = 2
settings.c = 3
settings.d = 4
settings.e = 5
settings.h = 6
settings.a.relation = "b + c"
settings.b.relation = "c"
settings.e.relation = "a"

sweeps = {"c": np.linspace(0, 1, 7), "d": np.linspace(-7, 0, 11)}
result = pyqsl.run(task, settings=settings, sweeps=sweeps)
