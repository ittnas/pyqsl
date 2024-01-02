"""
Simulates Ramsey experiment.

Antti Vepsalainen, 2019

Updated on 2023
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from qutip import *

import pyqsl

# A simple Ramsey experiment

settings = pyqsl.Settings()
settings.ts = 1e9  # Timescale of the simulation
settings.dw = 0e6  # Detuning in units of frequency
settings.dw.unit = "Hz"
settings.dw.relation = pyqsl.Equation(
    equation="dw * 2 * 3.1415 / ts"
)  # Convert to angular frequency and scale
settings.t1 = 1000e-9  # T1 time in seconds
settings.t1.unit = "s"
settings.t1.relation = pyqsl.Equation(equation="t1 * ts")
settings.tlist = np.linspace(0.0, settings.t1, 251)
settings.tlist.unit = "s"
settings.p0 = None
settings.p0.dimensions = ["tlist"]
settings.p1 = None
settings.p1.dimensions = ["tlist"]

sweep_arrays = {"dw": np.linspace(-10e6, 10e6, 51)}


def pre_processing_before_loop(settings):
    """preprocessing loop"""
    settings.tlist = np.linspace(0, settings.t1.value, 251)
    a = destroy(2)
    settings.H = settings.dw.value * a.dag() * a
    settings.psi0 = (basis(2, 0) + basis(2, 1)).unit()  # Initial state
    settings.output_list = [a + a.dag(), a.dag() * a]


def pre_processing_in_loop(settings):
    a = destroy(2)
    settings.H = settings.dw.value * a.dag() * a


def ramsey_simulation(H, psi0, tlist, output_list):
    output_temp = mesolve(H, psi0, tlist, [], output_list)
    output = {}
    for ii in range(len(output_list)):
        output["p" + str(ii)] = output_temp.expect[ii]
    return output


def add_tlist_as_sweep(settings, sweeps):
    return {
        f"p{ii}": {"tlist": settings.tlist.value}
        for ii in range(len(settings.output_list.value))
    }


def return_coordinate_for_tlist(settings, sweeps):
    return {settings.tlist, settings.tlist.value}


output = pyqsl.run(
    ramsey_simulation,
    settings=settings,
    sweeps=sweep_arrays,
    pre_process_before_loop=pre_processing_before_loop,
    pre_process_in_loop=pre_processing_in_loop,
    post_process_after_loop=add_tlist_as_sweep,
    parallelize=True,
)

output.save("ramsey_simulation.pickle")

##
plt.pcolor(output.tlist, output.dw, output.p0)
plt.show()
