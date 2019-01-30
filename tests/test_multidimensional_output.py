import numpy as np
#from pyqsl import core
import pyqsl.core as pyqsl
from qutip import *
import logging

params = {
    "p1":1,
    "p2":2,
    "trajs":list(range(6)),
    "tlist":list(range(7)),
}

sweep_arrays = {"p2":list(range(8))}

logging.basicConfig(level=logging.INFO)

output_list = []

for ii in range(len(sweep_arrays)):
    element = ('trajs',[])
    for jj in params['trajs']:
        element_l2 = ('tlist',[])
        for kk in params['tlist']:
            element_l3 = {'out1':([1,2,3+kk/(jj+1)],('x',[0.1,0.2,0.3])),'out2':kk + jj}
            #element_l3 = {'out1':([1,2,3+kk/(jj+1)],('x',[0.1,0.2,0.3])),'out2':([1,2,4,3+kk/(jj+1)],('x2',[0.1,0.2,0.3,0.4]))}
            element_l2[1].append(element_l3)
        element[1].append(element_l2)
        #('trajs',[('tlist',[{'out1':([1,2,3],('y',[0.1,0.2,0.3])},'out2':1.0}])])

    output_list.append(element)

pyqsl.save_data_hdf5("md_test",output_list,params,sweep_arrays,[],use_date_directory_structure=False,overwrite=True)
