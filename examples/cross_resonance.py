"""
Simulates a cross resonance gate between two qubits.

Antti Vepsalainen, 2019

Updated on 2023.
"""
import logging

logger = logging.getLogger(__name__)

import numpy as np
from qutip import *

import pyqsl
import matplotlib.pyplot as plt

w0 = 5.5  # QB0 freq
w1 = 6  # QB1 freq
a0 = 0.5  # Anharmonicity of QB0
a1 = 0.4  # Anharmonicity of QB1

g01 = 0.1  # QB 0-1 coupling
wd0 = 5  # Drive 0 frequency
wd1 = 4.6  # Drive 1 frequency
Ad0 = 0.05  # Drive 0 strength
Ad1 = 0.0  # Drive 1 strength
gamma_0 = 0.0  # QB0 relaxation rate
gamma_1 = 0.0  # QB1 relaxation rate

nq0 = 2  # Number of QB0 levels
nq1 = 2  # Number of QB1 levels

tlist = np.linspace(0, 2000, 501)  # Time instances in the simulation.

# The initial state.
psi0 = tensor((basis(nq0, 1) + basis(nq0, 0)).unit(), basis(nq1, 0))

# The list of output operators
output_list = [
    tensor(num(nq0), identity(nq1)),
    tensor(identity(nq0), num(nq1)),
    tensor(identity(nq0), create(nq1) + destroy(nq1)),
]

# These parameres are swept in the loop.
sweep_arrays = {"wd0": np.linspace(5.8, 6.2, 51)}

# This dictionary contains all the other parameters.
settings = pyqsl.Settings()
settings.w0 = w0
settings.w1 = w1
settings.a0 = a0
settings.a1 = a1
settings.g01 = g01
settings.wd0 = wd0
settings.wd0.unit = "GHz"
settings.wd1 = wd1
settings.wd1.unit = "GHz"
settings.Ad0 = Ad0
settings.Ad1 = Ad1
settings.nq0 = nq0
settings.nq1 = nq1
settings.gamma_0 = gamma_0
settings.gamma_1 = gamma_1
settings.tlist = tlist
settings.tlist.unit = "ns"
settings.psi0 = psi0
settings.output_list = output_list


def w0_coeff(t, args):
    """The coefficient containing the time dependent part of the Hamiltonian."""
    return np.sin(args["wd0"] * t)


def w1_coeff(t, args):
    """The coefficient containing the time dependent part of the Hamiltonian."""
    return np.sin(args["wd1"] * t)


def get_n_level_spec(w, a, n):
    """
    Get the energy-levels of an n-level transmon.

    The formula used: w_n = w*n - a*(n-1)
    """
    op = np.zeros((n, n))
    for ii in range(1, n):
        op[ii, ii] = w * ii - (ii - 1) * a
    return Qobj(op)


def get_empty(n):
    op = np.zeros((n, n))
    return Qobj(op)


def create_hamiltonian(s):
    """Prepares the Hamiltonian in the system in Qutip format. Needs to be recalculated at every step in the loop."""
    s.H = (
        tensor(
            get_n_level_spec(s.w0.value, s.a0.value, s.nq0.value),
            identity(s.nq1.value),
        )
        + tensor(
            identity(s.nq0.value),
            get_n_level_spec(s.w1.value, s.a1.value, s.nq1.value),
        )
        + s.g01.value
        * (
            tensor(create(s.nq0.value), destroy(s.nq1.value))
            + tensor(destroy(s.nq0.value), create(s.nq1.value))
        )
    )
    s.c_ops = [
        np.sqrt(s.gamma_0.value) * tensor(destroy(s.nq0.value), identity(s.nq1.value)),
        np.sqrt(s.gamma_1.value) * tensor(identity(s.nq0.value), destroy(s.nq1.value)),
    ]
    s.H1_0 = s.Ad0.value * tensor(
        create(s.nq0.value) + destroy(s.nq0.value), identity(s.nq1.value)
    )
    s.H1_1 = s.Ad1.value * tensor(
        identity(s.nq0.value), create(s.nq1.value) + destroy(s.nq1.value)
    )
    s.w0_coeff = w0_coeff
    s.w1_coeff = w1_coeff


def add_tlist_as_sweep(settings, sweeps):
    return {
        f"q{ii}": {"tlist": settings.tlist.value}
        for ii in range(len(settings.output_list.value))
    }


def task(H, H1_0, H1_1, w0_coeff, w1_coeff, psi0, tlist, c_ops, output_list, wd0, wd1):
    """
    Uses the qutip Lindblad equation solver to calcualte the time-evolution of the time-dependent Hamiltonian.
    """
    output_temp = mesolve(
        [
            H,
            [H1_0, w0_coeff],
            [H1_1, w1_coeff],
        ],
        psi0,
        tlist,
        c_ops,
        output_list,
        args={"wd0": wd0, "wd1": wd1},
    )
    output = {}
    for ii in range(len(output_list)):
        output["q" + str(ii)] = output_temp.expect[ii]
    return output


output = pyqsl.run(
    task,
    settings,
    sweeps=sweep_arrays,
    pre_process_in_loop=create_hamiltonian,
    post_process_after_loop=add_tlist_as_sweep,
    parallelize=True,
)
##
plt.pcolor(output.tlist, output.wd0, output.q0)
plt.colorbar()
plt.xlabel(f"{output.settings.tlist.name} ({output.settings.tlist.unit})")
plt.ylabel(f"{output.settings.wd0.name} ({output.settings.wd0.unit})")
plt.show()
##
output.save("cross_resonance.pickle")
