import numpy as np
#from pyqsl import core
import pyqsl.core as pyqsl
from qutip import *
import logging

# A simple Ramsey experiment

ts = 1e9
params = {
    "ts":ts,
    "dw":0e6 *2*np.pi / ts,
    "t1":1000e-9*ts
}

sweep_arrays = {"dw":np.linspace(-10e6 * 2*np.pi/ts,10e6 * 2*np.pi/ts,3)}

def pre_processing_before_loop(params,*args,**kwargs):
    #params["wd"] = params["wq"] + params["dw"]
    params["tlist"] = np.linspace(0,params["t1"],5)
    a = destroy(2)
    params["H"] = params["dw"]*a.dag()*a
    params["psi0"] = (basis(2,0) + basis(2,1)).unit() # Initial state
    params["output_list"] = [a + a.dag(),a.dag()*a]

def pre_processing_in_the_loop(params,*args,**kwargs):
    a = destroy(2)
    params["H"] = params["dw"]*a.dag()*a

def qubit_simulation_example(params,*args,**kwargs):
    output = mesolve(params["H"], params["psi0"], params["tlist"], [], params["output_list"])    
    return output


def qubit_simulation_example_labber(params,*args,**kwargs):
    """
    Work function suitable for use with Labber.
    """
    output_temp = mesolve(params["H"], params["psi0"], params["tlist"], [], params["output_list"])
    output = {}
    for ii in range(len(params["output_list"])):
        output['p' + str(ii)] = ("tlist",output_temp.expect[ii])
    return output


logging.basicConfig(level=logging.INFO)

output_list = pyqsl.simulation_loop(params,qubit_simulation_example_labber,sweep_arrays = sweep_arrays,pre_processing_before_loop = pre_processing_before_loop,pre_processing_in_the_loop = pre_processing_in_the_loop,parallelize=True)

pyqsl.save_data_hdf5("ramsey",output_list,params,sweep_arrays,[],use_date_directory_structure=False,overwrite=True)
