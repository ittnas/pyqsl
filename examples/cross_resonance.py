"""
Simulates a cross resonance gate between two qubits.

Antti Vepsalainen, 2019

Updated on 2023.
"""
import logging

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
from qutip import *

import pyqsl

settings = pyqsl.Settings()
settings.w0 = 5.0  # QB0 freq
settings.w0.unit = "GHz"
settings.w1 = 6.0  # QB1 freq
settings.w1.unit = "GHz"
settings.a0 = 0.5  # Anharmonicity of QB0
settings.a0.unit = "GHz"
settings.a1 = 0.4  # Anharmonicity of QB1
settings.a1.unit = "GHz"

settings.g01 = 0.25  # QB 0-1 coupling
settings.g01.unit = "GHz"
settings.wd0 = 5  # Drive 0 frequency
settings.wd0.unit = "GHz"
settings.wd0.relation = "w1 - g01**2/(w0 - w1)"  # Approximate frequency shift due to qubit-qubit coupling in the dispersive regime.
settings.wd1 = 4.6  # Drive 1 frequency
settings.wd1.unit = "GHz"
settings.Ad0 = 0.01  # Drive 0 strength
settings.Ad0.unit = "GHz"
settings.Ad1 = 0.0  # Drive 1 strength
settings.Ad1.unit = "GHz"
settings.gamma_0 = 0.0  # QB0 relaxation rate
settings.gamma_1 = 0.0  # QB1 relaxation rate

settings.nq0 = 2  # Number of QB0 levels
settings.nq1 = 2  # Number of QB1 levels

settings.tlist = np.linspace(0, 2000, 1001)  # Time instances in the simulation.
settings.tlist.unit = "ns"
settings.q0_t0 = 0
settings.q1_t0 = 0.5
settings.q0_mean = None
settings.q1_mean = None
settings.phase = None
settings.phase.dimensions = ["tlist"]
# The initial state.

# The list of output operators
settings.output_list = [
    tensor(num(settings.nq0.value), identity(settings.nq1.value)),
    tensor(identity(settings.nq0.value), num(settings.nq1.value)),
    tensor(
        identity(settings.nq0.value),
        create(settings.nq1.value) + destroy(settings.nq1.value),
    ),
    tensor(
        identity(settings.nq0.value),
        -1j * (-create(settings.nq1.value) + destroy(settings.nq1.value)),
    ),
]

for ii in range(len(settings.output_list.value)):
    setattr(settings, f"q{ii}", None)
    settings[f"q{ii}"].dimensions = ["tlist"]

# These parameres are swept in the loop.
sweeps = {
    # "Ad0": np.linspace(0.0, 0.2, 201),
    "q0_t0": [0, 1],
    # "wd0": np.linspace(5.9, 6.1, 21)
}


def w0_coeff(t, args):
    """The coefficient containing the time dependent part of the Hamiltonian."""
    return np.cos(args["wd0"] * t)


def w1_coeff(t, args):
    """The coefficient containing the time dependent part of the Hamiltonian."""
    return np.cos(args["wd1"] * t)


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
    s.psi0 = tensor(
        (
            basis(s.nq0.value, 1) * s.q0_t0.value
            + basis(s.nq0.value, 0) * (1 - s.q0_t0.value)
        ).unit(),
        (
            basis(s.nq1.value, 1) * s.q1_t0.value
            + basis(s.nq0.value, 0) * (1 - s.q1_t0.value)
        ).unit(),
    )


def task(
    H, H1_0, H1_1, w0_coeff, w1_coeff, psi0, tlist, c_ops, output_list, wd0, wd1, w0, w1
):
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
        output[f"q{ii}"] = output_temp.expect[ii]
        output[f"q{ii}_mean"] = np.mean(output_temp.expect[ii])
    output[f"phase"] = np.unwrap(
        np.angle(
            (np.real(output[f"q2"]) + 1j * np.real(output[f"q3"]))
            * np.exp(1j * wd0 * tlist)
        )
    )
    return output


output = pyqsl.run(
    task,
    settings,
    sweeps=sweeps,
    pre_process_in_loop=create_hamiltonian,
    parallelize=True,
)
##
fig, axs = plt.subplots(2, 2)
cr_op_point = np.argmax(
    (
        -np.abs(output.dataset.q1).isel(q0_t0=0)
        + np.abs(output.dataset.q1).isel(q0_t0=1)
    ).values
)
np.abs(output.dataset.q0).isel(q0_t0=0).plot(ax=axs[0, 0])
np.abs(output.dataset.q0).isel(q0_t0=1).plot(ax=axs[1, 0])
np.abs(output.dataset.q1).isel(q0_t0=0).plot(ax=axs[0, 1])
axs[0, 1].axvline(
    output.dataset.coords[output.dataset.q0.isel(q0_t0=1).dims[0]][cr_op_point],
    color="k",
)
np.abs(output.dataset.q1).isel(q0_t0=1).plot(ax=axs[1, 1])
axs[1, 1].axvline(
    output.dataset.coords[output.dataset.q0.isel(q0_t0=1).dims[0]][cr_op_point],
    color="k",
)
plt.show()
##
output.save("cross_resonance.pickle")
