import logging

import numpy as np
from qutip import *

# from pyqsl import core
import pyqsl.core as pyqsl

w0 = 5  # QB0 freq
w1 = 7  # QB1 freq
w2 = 4  # QB2 freq
g01 = 1  # QB 0-1 coupling
g12 = 1  # QB 1-2 coupling
g02 = 0  # QB 0-2 coupling
wd = 5  # Drive frequency
Ad = 0.001  # Drive strength
gamma_0 = 0.1  # QB0 relaxation rate
gamma_1 = 0.1  # QB1 relaxation rate
gamma_2 = 0.1  # QB2 relaxation rate

tlist = np.linspace(0, 100, 51)  # Time instances in the simulation.

psi0 = tensor(basis(2, 0), basis(2, 0), basis(2, 0))  # The initial state.

# The list of output operators
output_list = [
    tensor(sigmaz(), identity(2), identity(2)),
    tensor(identity(2), sigmaz(), identity(2)),
    tensor(identity(2), identity(2), sigmaz()),
]

# These parameres are swept in the loop.
sweep_arrays = {"w1": np.linspace(4, 8, 41), "wd": np.linspace(3, 9, 41)}

# This dictionary contains all the other parameters.
params = {
    "w0": w0,
    "w1": w1,
    "w2": w2,
    "g01": g01,
    "g12": g12,
    "g02": g02,
    "wd": wd,
    "Ad": Ad,
    "gamma_0": gamma_0,
    "gamma_1": gamma_1,
    "gamma_2": gamma_2,
    "tlist": tlist,
    "psi0": psi0,
    "output_list": output_list,
}


def H1_coeff(t, args):
    """The coefficient containing the time dependent part of the Hamiltonian."""
    return np.sin(args["wd"] * t)


def create_hamiltonian(params, *args, **kwargs):
    """Prepares the Hamiltonian in the system in Qutip format. Needs to be recalculated at every step in the loop."""
    params["H"] = (
        0.5
        * (
            tensor(params["w0"] * sigmaz(), identity(2), identity(2))
            + tensor(identity(2), sigmaz() * params["w1"], identity(2))
            + tensor(identity(2), identity(2), sigmaz() * params["w2"])
        )
        + params["g01"] * tensor(sigmax(), sigmax(), identity(2))
        + params["g12"] * tensor(identity(2), sigmax(), sigmax())
        + params["g02"] * tensor(sigmax(), identity(2), sigmax())
    )
    params["c_ops"] = [
        np.sqrt(params["gamma_0"]) * tensor(destroy(2), identity(2), identity(2)),
        np.sqrt(params["gamma_1"]) * tensor(identity(2), destroy(2), identity(2)),
        np.sqrt(params["gamma_2"]) * tensor(identity(2), identity(2), destroy(2)),
    ]
    params["H1"] = (
        params["Ad"] * tensor(sigmax(), identity(2), identity(2))
        + tensor(identity(2), sigmax(), identity(2))
        + tensor(identity(2), identity(2), sigmax())
    )
    params["H1_coeff"] = H1_coeff
    # print(params["H"])


def task(params, *args, **kwargs):
    """The simulation task"""
    output_temp = mesolve(
        [params["H"], [params["H1"], params["H1_coeff"]]],
        params["psi0"],
        params["tlist"],
        params["c_ops"],
        params["output_list"],
        args={"wd": params["wd"]},
    )
    output = {}
    for ii in range(len(params["output_list"])):
        output["q" + str(ii)] = (output_temp.expect[ii], ("tlist", params["tlist"]))
    return output


output = pyqsl.simulation_loop(
    params,
    task,
    sweep_arrays=sweep_arrays,
    pre_processing_in_the_loop=create_hamiltonian,
    parallelize=True,
)
pyqsl.save_data_hdf5(
    "three_qubits_1",
    output,
    params,
    sweep_arrays,
    [],
    use_date_directory_structure=False,
    overwrite=False,
)
