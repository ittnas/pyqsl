import numpy as np
import copy as copy
import inspect
import os
import datetime
from qutip import *
import matplotlib.pyplot as plt
import sys
import json
import pickle
#import noise_generator as ng
import traceback
import collections
sys.path.insert(0, '/usr/share/Labber/Script')
import Labber

def default_save_element_fun(save_path,output,ii):
    qsave(output,os.path.join(save_path,'qobject_' + str(ii)))

def default_save_parameters_function(full_save_path,params,sweep_arrays,derived_arrays):
    #print('In the default save elements fun.')
    with open(os.path.join(full_save_path,'parameters.json'),'w') as f:
        try:
            json.dump(params,f)
        except Exception:
            print('Unable to dump parameters to a file. Parameters are not saved.')
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)


    with open(os.path.join(full_save_path,'sweep_arrays.json'),'w') as f:
        sweep_arrays_s = {}
        for key,value in sweep_arrays.items():
            # XXX Not a proper way to save a list to json.
            sweep_arrays_s[key] = str(value)
        try:
            json.dump(sweep_arrays_s,f)
        except Exception:
            print('Unable to dump sweep_arrays to a file. Sweep arrays are not saved.')
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)


    with open(os.path.join(full_save_path,'derived_arrays.json'),'w') as f:
        derived_arrays_s = {}
        for key,value in derived_arrays.items():
            # XXX Not a proper way to save a dict to json
            derived_arrays_s[key] = str(value)
        try:
            json.dump(derived_arrays_s,f)
        except Exception:
            print('Unable to dump derived_arrays to a file. Derived arrays are not saved.')
            print('-'*60)
            traceback.print_exc(file=sys.stdout)
            print('-'*60)

def default_save_data_function(save_path,sweep_arrays,derived_arrays,output_array,save_element_function):
    try:
        for ii,output in enumerate(output_array):
            save_element_function(save_path,output,ii)
    except Exception:
        print('Error in the save_element_fun. Data cannot be saved. The error is: ', sys.exc_info()[1])

def simulation_loop(params,simulation_task,sweep_arrays = {},derived_arrays = {},pre_processing_before_loop = None,pre_processing_in_the_loop = None, post_processing_in_the_loop = None, parallelize=False):
    """
    This is the main simulation loop.

    Parameters
    --------
    params : dict
        A dictionary containing the simulation parameters. The key is a string giving the name of the parameter
    simulation_task : function handle
        A function that performs the simulation. Should have form [output = simulation_task(params_private)], where output
        is the result of the simulation.
    """
    start_time = datetime.datetime.now()
    print('Simulation started at ' + str(start_time))
    dims = []
    for key,value in sweep_arrays.items():
        dims.append(len(value))

    print(dims)
    if dims ==[]:
        dims = 1
    N_tot = np.prod(dims)
    print('Sweep dimensions: ' + str(dims) + '.')
    output_array = [None]*N_tot

    if(pre_processing_before_loop):
        pre_processing_before_loop(params)

    for ii in range(N_tot):
        # The master loop
        params_private = copy.deepcopy(params) # Make sure that parallel threads don't simulataneously edit params. Only use params_private in the following
        current_ind = np.unravel_index(ii,dims)
        sweep_array_index = 0
        for key,value in sweep_arrays.items():
             # Update all the parameters
             params_private[key] = sweep_arrays[key][current_ind[sweep_array_index]]
             sweep_array_index = sweep_array_index + 1
             #print(params_private)

        # Update the paremeters based on the derived arrays
        derived_arrays_index = 0
        for key,value in derived_arrays.items():
            for subkey, subvalue in value.items():
                # Update all the parameters
                params_private[subkey] = derived_arrays[key][subkey][current_ind[derived_arrays_index]]
                #[current_ind[sweep_array_index]]
            #print(params_private)
            derived_arrays_index = derived_arrays_index + 1


        if(pre_processing_in_the_loop):
            pre_processing_in_the_loop(params_private)
        
        # params_private now contains all the required information to run the simulation
        output = simulation_task(params_private)
        if(post_processing_in_the_loop):
            output = post_processing_in_the_loop(output,params)
        output_array[ii] = output
    end_time = datetime.datetime.now()
    print('Simulation finished at ' + str(end_time) + '. The duration of the simulation was ' + str(end_time-start_time) + '.')
    return output_array

def save_data(save_path,output_array,params,sweep_arrays,derived_arrays,save_element_fun = default_save_element_fun, save_parameters_function = default_save_parameters_function, save_data_function = default_save_data_function, use_date_directory_structure = True):
    """ Saves the data to a directory given by save_path/data/current_date if use_date_directory_structure is True. Otherwise saves the data to save_path/.

    Parameters
    ----------
    save_path : str
        Full path to which data is saved.
    ouput_array : array_like
        An array containing the output of the simulation. Should have dimensions n1*n2*n3...ni*nbr_outputs, where i is the number of dimensions in the simulation and nbr_outputs is the number of elements in the output.
    params : dict
        A dictionary containing the simulation parameters.
    sweep_arrays : dict
        A dictionary containing the parameters that are being swept as keys and arrays of swept parameters as values.
    derived_arrays : dict
        A dictionary containing dictionaries of parameters that are related to parameters in sweep_arrays
    save_element_fun : function handle, optional
        A function containing instructions for saving a single element in the output_array. Needs to have format save_element_fun(save_path,output,ii), where save_path is the path to which data is saved, output is a
        single elmenet of output array and ii is the index of the serialized element.
    save_parameters_fun : function handle, optional
        A function used to save the parameters dictionary. Has the format save_parameters_fun(full_save_path,params,sweep_arrays,derived_arrays)
    use_date_directory_structure : bool, optional
       If True, add date structure to save_path, otherwise save directly to save_path.
    """
    if use_date_directory_structure:
        full_save_path = os.path.join(save_path,'data',str(datetime.datetime.now())) # XXX I don't like the date format. Change it when python documentation is available.
    else:
        full_save_path = save_path
    try:
        os.makedirs(full_save_path)
    except FileExistsError:
        # path already exists, just continue
        pass
    except OSError as err:
        print('Error while creating the saving directory. Data cannot be saved. The error is:',sys.exc_info()[1])


    # Saving parameters
    save_parameters_function(full_save_path,params,sweep_arrays,derived_arrays)    
    save_data_function(full_save_path,sweep_arrays,derived_arrays,output_array,save_element_fun)
    return full_save_path

def save_data_hdf5(filename,data_array,params,sweep_arrays,derived_arrays, use_date_directory_structure = True, overwrite = False,save_path = ''):
    """ Saves the simulation data to hdf5 file.

    Parameters
    ----------
    filename : str
        The name of the log file.
    data_array : list of dicts or list of dicts of dicts
        A list which size equal to the total number of elements in the simulation.
        Each element of the list is a dictionary with the name of the data variable as the key and 
        the value of the element as the value. If the simulation result contains vector data, the value of an element
        can be another dictionary which keys are the names of the vector variables and values are arrays containing the data. If the name of the vector variable is found in the params dictionary, it has to have the same number of elements as is the length of the data vector.
    sweep_arrays : dict of lists
        A dictionary which keys are the names of the swept parameters and the values are the arrays which are swept.
    derived_arrays : dict of dict of list
        A dictionary which keys are the names of the variables from which the arrays are derived. The names have also to be keys in the sweep_arrays variable. Values of the dictionary are an other dictionary which keys are names of the derived arrays and values are arrays according to which the parameter is varied. The array has to have length equal to the corresponding parameter in the sweep_arrays.
    use_data_directory_structure : bool
        Indicates whether the data should be saved in a directory structure containing the current date. Has to be True if Labber databases are to be used. Currently Labber automatically uses a main directory which can only be set from the LogBrowser program (or config file).
    overwrite : bool
        If True, previous simulation data with the same name will be overwritten. Otherwise will append _n to the end of the filename if the file with the same name already exists.
    save_path : str
        The path to which the data should be saved. Currently it is not possible to easily set in Labber, and thus it is recommended not to use this variable.

    Returns
    -------
    str
        The full save path including the name of the file the data is saved to.
        
    """
    dims = []
    for key,value in sweep_arrays.items():
        dims.append(len(value))
    if dims ==[]:
        dims = 1
    N_tot = np.prod(dims)


# Creating step channels
    temp_ouput_array = {}
    data_dicts = []
    for key,value in data_array[0].items():
        temp_ouput_array[key]=[None]*N_tot
        data_dicts.append(dict(name=key,unit='',vector=False))
    for ii in range(len(data_array)):
        for key,value in data_array[ii].items():
            temp_ouput_array[key][ii] = value
    data_array = temp_ouput_array
    channels = []
    for array_name, array_values in sweep_arrays.items():
        channels.append(dict(name=array_name,unit='',values=array_values))
    vector_in_the_data = False

    first_output_element = list(data_array.values())[0][0]
    if isinstance(first_output_element, tuple):
        vector_in_the_data = True
        vector_channel_name = first_output_element[0]
        if vector_channel_name in params:
            vector_channel_values = params[vector_channel_name]
        else:
            vector_channel_values = range(len(first_output_element[1]))
        channels.append(dict(name=vector_channel_name,unit='',values=vector_channel_values))

    channels.reverse() # Reversing is important to make the inner and outer looping consistent.
    
# And then the channels for the rest of the parameters
    log_channels = []
    log_channels.extend(channels)
    for param_name, param_value in params.items():
        try:
            if(param_name not in sweep_arrays and param_name != vector_channel_name):
                if not (isinstance(param_value,(float,int,str))):
                    raise ValueError
                channel_dict =dict(name=param_name,unit='',values=param_value,vector=False)
                #print(channel_dict)
                log_channels.append(channel_dict)
        except ValueError:
            print('Parameter ' + param_name + ' cannot be saved.')
            pass

    unique_filename = os.path.join(save_path,filename)
    

    if not overwrite:
        f_temp = Labber.createLogFile_ForData(os.path.join(save_path,'dummy_file_name'),'')
        filename_temp = f_temp.getFilePath('')
        os.remove(filename_temp)
        print(filename_temp)
        dirname = os.path.dirname(filename_temp)
        unique_filename = os.path.join(save_path,filename)
        ii=1
        print(os.path.join(dirname,unique_filename + '.hdf5'))
        print(os.path.isfile(os.path.join(dirname,unique_filename + '.hdf5')))
        while os.path.isfile(os.path.join(dirname,unique_filename + '.hdf5')):
           
           unique_filename = os.path.join(save_path,filename + '_' + str(ii))
           print(os.path.join(dirname,unique_filename))
           ii = ii + 1
           if ii > 100:
               print('Could not create unique filename for ' + filename)
               return ''

    print(os.path.join(save_path,unique_filename))
    f = Labber.createLogFile_ForData(os.path.join(save_path,unique_filename),data_dicts,log_channels,use_database= not use_date_directory_structure)

    channel = channels[0]
    jj = 0
    n_lowest = len(channel['values'])
    print(n_lowest)
    if vector_in_the_data:
        skip = 1
    else:
        skip = n_lowest
    data_dict = {}
    for ii in range(0,N_tot,skip):
        data_dict = {channel['name'] : channel['values']}
        for key, values in data_array.items():
            #data_dict = {channel['name'] : channel['values'],key: np.array(values[ii:ii+n_lowest])}
            if vector_in_the_data:
                data_dict[key] = np.array(values[ii][1])
            else:
                data_dict[key] = np.array(values[ii:ii+n_lowest])
        f.addEntry(data_dict)


        
    return f.getFilePath([])

