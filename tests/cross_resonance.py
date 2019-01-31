"""
Simulates a cross resonance gate between two qubits.

Antti Vepsalainen, 2019
"""

import numpy as np
# from pyqsl import core
import pyqsl.core as pyqsl
from qutip import *
import logging

w0 = 5.5  # QB0 freq
w1 = 6  # QB1 freq
a0 = 0.5  # Anharmonicity of QB0
a1 = 0.4  # Anharmonicity of QB1

g01 = 0.2  # QB 0-1 coupling
wd0 = 5  # Drive 0 frequency
wd1 = 4.6  # Drive 1 frequency
Ad0 = 0.05  # Drive 0 strength
Ad1 = 0.0  # Drive 1 strength
gamma_0 = 0.0  # QB0 relaxation rate
gamma_1 = 0.0  # QB1 relaxation rate

nq0 = 2  # Number of QB0 levels
nq1 = 2  # Number of QB1 levels

tlist = np.linspace(0, 500, 101)  # Time instances in the simulation.

psi0 = tensor(basis(nq0, 1), basis(nq1, 0))  # The initial state.

# The list of output operators
output_list = [tensor(num(nq0), identity(nq1)),
               tensor(identity(nq0), num(nq1))]

# These parameres are swept in the loop.
sweep_arrays = {"wd0": np.linspace(5.8, 6.2, 511)}

# This dictionary contains all the other parameters.
params = {
    "w0": w0,
    "w1": w1,
    "a0": a0,
    "a1": a1,
    "g01": g01,
    "wd0": wd0,
    "wd1": wd1,
    "Ad0": Ad0,
    "Ad1": Ad1,
    "nq0": nq0,
    "nq1": nq1,
    "gamma_0": gamma_0,
    "gamma_1": gamma_1,
    "tlist": tlist,
    "psi0": psi0,
    "output_list": output_list,
}


def w0_coeff(t, args):
    """ The coefficient containing the time dependent part of the Hamiltonian. """
    return np.sin(args['wd0']*t)


def w1_coeff(t, args):
    """ The coefficient containing the time dependent part of the Hamiltonian. """
    return np.sin(args['wd1']*t)


def get_n_level_spec(w, a, n):
    """
    Get the energy-levels of an n-level transmon.

    The formula used: w_n = w*n - a*(n-1)
    """
    op = np.zeros((n, n))
    for ii in range(1, n):
        op[ii, ii] = w*ii - (ii-1)*a
    return Qobj(op)


def get_empty(n):
    op = np.zeros((n, n))
    return Qobj(op)


def create_hamiltonian(params, *args, **kwargs):
    """ Prepares the Hamiltonian in the system in Qutip format. Needs to be recalculated at every step in the loop."""
    params["H"] = tensor(get_n_level_spec(
        params["w0"], params["a0"], params["nq0"]), identity(params['nq1'])) + tensor(identity(params['nq0']), get_n_level_spec(params["w1"], params["a1"], params["nq1"])) + params["g01"]*(tensor(create(params['nq0']), destroy(params['nq1'])) + tensor(destroy(params['nq0']), create(params['nq1'])))
    params["c_ops"] = [np.sqrt(params["gamma_0"])*tensor(destroy(params["nq0"]), identity(
        params["nq1"])), np.sqrt(params["gamma_1"])*tensor(identity(params["nq0"]), destroy(params["nq1"]))]
    params["H1_0"] = params["Ad0"] * \
        tensor(create(params['nq0']) +
               destroy(params['nq0']), identity(params['nq1']))
    params["H1_1"] = params["Ad1"] * \
        tensor(identity(params['nq0']), create(
            params['nq1'])+destroy(params['nq1']))
    params["w0_coeff"] = w0_coeff
    params["w1_coeff"] = w1_coeff


def task(params, *args, **kwargs):
    """
    Uses the qutip Lindblad equation solver to calcualte the time-evolution of the time-dependent Hamiltonian.
    """
    output_temp = mesolve([params["H"], [params["H1_0"], params["w0_coeff"]], [params["H1_1"], params["w1_coeff"]]], params["psi0"],
                          params["tlist"], params["c_ops"], params["output_list"], args={"wd0": params["wd0"], "wd1": params["wd1"]})
    output = {}
    for ii in range(len(params["output_list"])):
        output['q' + str(ii)] = (output_temp.expect[ii],
                                 ('tlist', params['tlist']))
    return output


output = pyqsl.simulation_loop(params, task, sweep_arrays=sweep_arrays,
                               pre_processing_in_the_loop=create_hamiltonian, parallelize=False)
pyqsl.save_data_hdf5("cross_resonance", output, params, sweep_arrays, [


], use_date_directory_structure=False, overwrite=True)
