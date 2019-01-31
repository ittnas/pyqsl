from packaging import version as pkg
from functools import partial
import multiprocessing as mp
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
import traceback
import collections
import logging
sys.path.insert(0, '/usr/share/Labber/Script')  # For Labber


def _default_save_element_fun(save_path, output, ii):
    """ Saves the element using qutip qsave function

    Parameters
    ----------
    save_path : str
        The path to which the data is saved.
    output : obj
        The object to save.
    ii : int
        The serial index of the output element in the simulation.
    """
    qsave(output, os.path.join(save_path, 'qobject_' + str(ii)))


def _default_save_parameters_function(full_save_path, params, sweep_arrays, derived_arrays):
    with open(os.path.join(full_save_path, 'parameters.json'), 'w') as f:
        try:
            json.dump(params, f)
        except Exception as e:
            logging.warning(
                'Unable to dump parameters to a file. Parameters are not saved.')
            logging.warning(e, exc_info=True)
            #print('Unable to dump parameters to a file. Parameters are not saved.')
            # print('-'*60)
            # traceback.print_exc(file=sys.stdout)
            # print('-'*60)

    with open(os.path.join(full_save_path, 'sweep_arrays.json'), 'w') as f:
        sweep_arrays_s = {}
        for key, value in sweep_arrays.items():
            # XXX Not a proper way to save a list to json.
            sweep_arrays_s[key] = str(value)
        try:
            json.dump(sweep_arrays_s, f)
        except Exception as e:
            logging.warning(
                'Unable to dump sweep_arrays to a file. Sweep arrays are not saved.')
            logging.warning(e, exc_info=True)
            #print('Unable to dump sweep_arrays to a file. Sweep arrays are not saved.')
            # print('-'*60)
            # traceback.print_exc(file=sys.stdout)
            # print('-'*60)

    with open(os.path.join(full_save_path, 'derived_arrays.json'), 'w') as f:
        derived_arrays_s = {}
        for key, value in derived_arrays.items():
            # XXX Not a proper way to save a dict to json
            derived_arrays_s[key] = str(value)
        try:
            json.dump(derived_arrays_s, f)
        except Exception as e:
            logging.warning(
                'Unable to dump derived_arrays to a file. Derived arrays are not saved.')
            logging.warning(e, exc_info=True)

            #print('Unable to dump derived_arrays to a file. Derived arrays are not saved.')
            # print('-'*60)
            # traceback.print_exc(file=sys.stdout)
            # print('-'*60)


def _default_save_data_function(save_path, sweep_arrays, derived_arrays, output_array, save_element_function):
    try:
        for ii, output in enumerate(output_array):
            save_element_function(save_path, output, ii)
    except Exception as e:
        #print('Error in the save_element_fun. Data cannot be saved. The error is: ', sys.exc_info()[1])
        logging.warning('Error in the save_element_fun. Data cannot be saved.')
        logging.warning(e, exc_info=True)


def _simulation_loop_body(ii, params, dims, sweep_arrays, derived_arrays, pre_processing_in_the_loop, post_processing_in_the_loop, simulation_task):
    # The master loop
    # Make sure that parallel threads don't simulataneously edit params. Only use params_private in the following
    params_private = copy.deepcopy(params)
    current_ind = np.unravel_index(ii, dims)
    sweep_array_index = 0
    for key, value in sweep_arrays.items():
        # Update all the parameters
        params_private[key] = sweep_arrays[key][current_ind[sweep_array_index]]
        sweep_array_index = sweep_array_index + 1
        # print(params_private)

    # Update the paremeters based on the derived arrays
    derived_arrays_index = 0
    for key, value in derived_arrays.items():
        for subkey, subvalue in value.items():
            # Update all the parameters
            params_private[subkey] = derived_arrays[key][subkey][current_ind[derived_arrays_index]]
            # [current_ind[sweep_array_index]]
        # print(params_private)
        derived_arrays_index = derived_arrays_index + 1

    if(pre_processing_in_the_loop):
        pre_processing_in_the_loop(params_private)

        # params_private now contains all the required information to run the simulation
    output = simulation_task(params_private)

    if(post_processing_in_the_loop):
        output = post_processing_in_the_loop(output, params)
    return output


def simulation_loop(params, simulation_task, sweep_arrays={}, derived_arrays={}, pre_processing_before_loop=None, pre_processing_in_the_loop=None, post_processing_in_the_loop=None, parallelize=False):
    """
    This is the main simulation loop.

    Parameters
    ----------
    params : dict
        A dictionary containing the simulation parameters. The key is a string giving the name of the parameter.
    simulation_task : function handle
        A function that performs the simulation. Should have form [output = simulation_task(params)], where output
        is the result of the simulation.
    sweep_arrays : dict, opt
        A dictionary containing the parameters that are being swept as keys and arrays of swept parameters as values.
    derived_arrays : dict, opt
        A dictionary containing dictionaries of parameters that are related to parameters in sweep_arrays
    pre_processing_before_loop : function handle, opt
        Function to pre-process the parameter array. Takes params dictionary as an input.
    pre_processing_in_the_loop : function handle, opt
        Modifies the parameter array in the loop. All the parameters that are dependant on the swept parameters should be recalculated here.
    post_processing_in_the_loop : function handle, opt
        Function can be used to modify the output of the simulation task. Takes params as an input.
    parallelize : bool, opt
        Boolean indicating whether the computation should be parallelized.

    """
    start_time = datetime.datetime.now()
    logging.info('Simulation started at ' + str(start_time))
    #print('Simulation started at ' + str(start_time))
    dims = []
    for key, value in sweep_arrays.items():
        dims.append(len(value))

    logging.debug(dims)
    if dims == []:
        dims = 1
    N_tot = np.prod(dims)
    #print('Sweep dimensions: ' + str(dims) + '.')
    logging.info('Sweep dimensions: ' + str(dims) + '.')
    output_array = [None]*N_tot

    if(pre_processing_before_loop):
        pre_processing_before_loop(params)

    simulation_loop_body_partial = partial(_simulation_loop_body, params=params, dims=dims, sweep_arrays=sweep_arrays, derived_arrays=derived_arrays,
                                           pre_processing_in_the_loop=pre_processing_in_the_loop, post_processing_in_the_loop=post_processing_in_the_loop, simulation_task=simulation_task)
    if parallelize:
        with mp.Pool(processes=None) as p:
            output_array = p.map(simulation_loop_body_partial, range(N_tot))
    else:
        for ii in range(N_tot):
            output = simulation_loop_body_partial(ii)
            output_array[ii] = output
    end_time = datetime.datetime.now()
    logging.info('Simulation finished at ' + str(end_time) +
                 '. The duration of the simulation was ' + str(end_time-start_time) + '.')
    #print('Simulation finished at ' + str(end_time) + '. The duration of the simulation was ' + str(end_time-start_time) + '.')
    return output_array


def save_data(save_path, output_array, params, sweep_arrays, derived_arrays, save_element_fun=_default_save_element_fun, save_parameters_function=_default_save_parameters_function, save_data_function=_default_save_data_function, use_date_directory_structure=True):
    """ Saves the data to a directory given by save_path/data/current_date if use_date_directory_structure is True. Otherwise saves the data to save_path/. DEPRACATED.

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
        # XXX I don't like the date format. Change it when python documentation is available.
        full_save_path = os.path.join(
            save_path, 'data', str(datetime.datetime.now()))
    else:
        full_save_path = save_path
    try:
        os.makedirs(full_save_path)
    except FileExistsError:
        # path already exists, just continue
        pass
    except OSError as err:
        logging.error(
            'Error while creating the saving directory. Data cannot be saved.')
        logging.error(err, exc_info=True)
        #print('Error while creating the saving directory. Data cannot be saved. The error is:',sys.exc_info()[1])

    # Saving parameters
    save_parameters_function(full_save_path, params,
                             sweep_arrays, derived_arrays)
    save_data_function(full_save_path, sweep_arrays,
                       derived_arrays, output_array, save_element_fun)
    return full_save_path


def save_data_hdf5(filename, data_array, params, sweep_arrays, derived_arrays, use_date_directory_structure=True, overwrite=False, save_path='', comment=None, tags=[], project=None, user=None):
    """ Saves the simulation data to hdf5 file.

    Parameters
    ----------
    filename : str
        The name of the log file.
    data_array : list of dicts or a nested list of tuples
        A list which size equal to the total number of elements in the simulation.
        The elements may either contain a dictionary (its structure is explained later), that contains the data variables or a tuple that corresponds to additional dimension in the data array. The first element of the tuple is the name of the dimension and the second element is a list that contains its data elements. If the dimension name is found in the params, its x-axis is given by the vector params[dimension_name]. Effectively, the data dimension is treated as a new element in the sweep_arrays. The list - tuple pairs can be nested arbitrarily many times.

        The keys of the data dictionary are the names of the data variables. The values are either scalars or tuples with the following shape: the first element is the data vector and the second element is a tuple with the first element being the name of the x-axis for the data vector and the second element is the x-axis values for the data.

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
    tags : list of str, opt
        List of tags to attach to the log.
    project : str, opt
        Name of the project.
    user : str, opt
        The name of the user who created the log.

    Returns
    -------
    str
        The full save path including the name of the file the data is saved to.

    """
    try:
        import Labber  # import Labber only if it is needed.
    except ImportError as err:
        logging.error(
            'Labber script tools could not be found. Data is not saved.')
        logging.error(err, exc_info=True)
        #print('Labber script tools could not be found. Data is not saved.')
        return None
    #print(version.parse(Labber.__version__) < version.parse('1.6.1'))
    # It is actually 1.6.2 that is required, but comparing to it returns True.
    if pkg.parse(Labber.__version__) < pkg.parse('1.6.1'):
        logging.error('Labber version at least 1.6.2 is required. Current version is ' +
                      str(Labber.__version__) + '. Data cannot be saved.')
        return
    dims = []
    for key, value in sweep_arrays.items():
        dims.append(len(value))
    if dims == []:
        dims = 1
    N_tot = np.prod(dims)

    # Creating data channels. In order to be compatible with Labber, the data needs to be rearranged from
    # output_array
    #   . dict
    #         data_name : data_values
    #         .
    #         .
    #         .
    #   . dict

    # temp_ouput_array = {}
    # data_dicts = []
    # for key,value in data_array[0].items():
    #     temp_ouput_array[key]=[None]*N_tot
    #     data_dicts.append(dict(name=key,unit='',vector=False))
    # for ii in range(len(data_array)):
    #     for key,value in data_array[ii].items():
    #         temp_ouput_array[key][ii] = value
    # data_array = temp_ouput_array

# Creating step channels
    step_channels = []
    step_channel_names = []
    for array_name, array_values in sweep_arrays.items():
        step_channels.append(
            dict(name=array_name, unit='', values=array_values))
        step_channel_names.append(array_name)
    vector_in_the_data = False

    # first_output_element = list(data_array.values())[0][0]
    # print(data_array.values())
    # if isinstance(first_output_element, tuple):
    #     vector_in_the_data = True
    #     vector_channel_name = first_output_element[0]
    #     if vector_channel_name in params:
    #         vector_channel_values = params[vector_channel_name]
    #     else:
    #         vector_channel_values = range(len(first_output_element[1]))
    #     step_channels.append(dict(name=vector_channel_name,unit='',values=vector_channel_values))

    first_output_element = data_array[0]
    #print("First output element.")
    # print(first_output_element)
    current_element = first_output_element
    while isinstance(current_element, tuple):
        vector_in_the_data = True  # Not sure if this is needed.
        # Found a tuple. The format should be ('name','value'), where value can be another tuple or then a dict, which keys are the log channel names and values are the data. The data can either be a scalar or a tuple (data_values,('x_axis_parameter_name',x_axis_values))
        new_step_channel_name = current_element[0]
        # The tuple 'name' is in the params. Look for its dimensions from there.
        if new_step_channel_name in params:
            new_step_channel_values = params[new_step_channel_name]
        else:  # It's not in params. Therefore create a simple index array as the step channel values
            # This might be risky, if current_element[1] is not a list. Though it certainly should be.
            new_step_channel_values = list(range(len(current_element[1])))

        #print('New step channel values')
        # print(new_step_channel_values)
        step_channels.append(
            dict(name=new_step_channel_name, values=new_step_channel_values))
        step_channel_names.append(new_step_channel_name)
        # The first element of the next element.
        current_element = current_element[1][0]

    # Now current element should be a dict that contains the data chanels.
    data_channels = []
    # Found a dict! The keys are the data channels.
    if isinstance(current_element, dict):
        for data_name, data_value in current_element.items():
            if isinstance(data_value, tuple):
                x_name = data_value[1][0]
                #x_values = data_value[1][1]
                data_channels.append(dict(name=data_name, x_name=x_name))
            # else if isinstance(data_value,(list,np.ndarray)):
            else:
                data_channels.append(dict(name=data_name))
    else:
        logging.error(
            'No data channels found in the data. Either data is incorrectly formatted or there is nothing to save')
        return None
    # Reversing is important to make the inner and outer looping consistent.
    step_channels.reverse()
    # print(step_channels)
    # XXX log_channels name is confusing, as this is going to be appended to step_channels dict.
    log_channels = []
    #log_dict = {}
    log_channels.extend(step_channels)
    for param_name, param_value in params.items():
        try:
            if(param_name not in sweep_arrays and param_name not in step_channel_names):
                if not (isinstance(param_value, (float, int, bool))):
                    raise ValueError  # Comment this line you you figure out a way to save strings
                    try:
                        param_value_str = str(param_value)
                        # if the objects string format is too long, don't include it.
                        if len(param_value_str) > 1e5:
                            raise ValueError
                        param_value = param_value_str
                    except Error:  # Should be replaced with the actual errors that str() can raise.
                        raise ValueError
                channel_dict = dict(name=param_name, unit='',
                                    values=param_value, vector=False)
                # print(channel_dict)
                log_channels.append(channel_dict)
                #log_dict[channel_dict["name"]] = channel_dict["values"]
        except ValueError:
            logging.warning('Parameter ' + param_name + ' cannot be saved.')
            #print('Parameter ' + param_name + ' cannot be saved.')
            pass

    unique_filename = os.path.join(save_path, filename)
    max_number_of_attempts = 100

    if not overwrite:
        # Get the path to which Labber tries to create the log by creating a dummy log.
        # Not very neat!
        f_temp = Labber.createLogFile_ForData(
            os.path.join(save_path, 'dummy_file_name'), '')
        filename_temp = f_temp.getFilePath('')
        os.remove(filename_temp)
        logging.debug(filename_temp)
        dirname = os.path.dirname(filename_temp)
        unique_filename = os.path.join(save_path, filename)
        ii = 1
        while os.path.isfile(os.path.join(dirname, unique_filename + '.hdf5')):
            # Attempts to find a filename that is not used.
            unique_filename = os.path.join(save_path, filename + '_' + str(ii))
            print(os.path.join(dirname, unique_filename))
            ii = ii + 1
            if ii > max_number_of_attempts:
                logging.warning(
                    'Could not create unique filename for ' + filename + '.')
                return ''

    logging.debug(os.path.join(save_path, unique_filename))
    # Finally create the real log file.

    #f = Labber.createLogFile_ForData(os.path.join(save_path,unique_filename),data_dicts,log_channels,use_database= not use_date_directory_structure)
    f = Labber.createLogFile_ForData(os.path.join(
        save_path, unique_filename), data_channels, log_channels, use_database=not use_date_directory_structure)

    if len(tags) > 0:
        f.setTags(tags)
    if project:
        f.setProject(project)
    if user:
        f.setUser(user)
    if comment:
        f.setComment(comment)

    # channel = step_channels[0]
    # jj = 0
    # n_lowest = len(channel['values'])
    # logging.debug(n_lowest)
    # if vector_in_the_data:
    #     skip = 1
    # else:
    #     skip = n_lowest
    # data_dict = {}

    def add_entries(element, f):
        """
        Recursively adds the entries to labber dict f given by element.
        """
        if isinstance(element, tuple):
            for element_l2 in element[1]:
                add_entries(element_l2, f)
        else:
            data_dicts = {}
            for data_name, data_value in element.items():
                #print(data_name, data_value)
                if isinstance(data_value, tuple):
                    x_values = data_value[1][1]
                    data_dict = Labber.getTraceDict(data_value[0], x=x_values)
                    data_dicts[data_name] = data_dict
                else:
                    data_dict = Labber.getTraceDict(data_value)
                    data_dicts[data_name] = data_dict
            # print('data_dicts')
            # print(data_dicts)
            f.addEntry(data_dicts)

    for ii in range(0, N_tot, 1):
        add_entries(data_array[ii], f)

    # for ii in range(0,N_tot,skip):
    #     data_dict = {channel['name'] : channel['values']}
    #     for key, values in data_array.items():
    #         #data_dict = {channel['name'] : channel['values'],key: np.array(values[ii:ii+n_lowest])}
    #         if vector_in_the_data:
    #             data_dict[key] = np.array(values[ii][1])
    #         else:
    #             data_dict[key] = np.array(values[ii:ii+n_lowest])

    #     #data_dict.update(log_dict)
    #     f.addEntry(data_dict)

    final_path = f.getFilePath([])
    logging.info('Data is saved to ' + final_path + '.')
    return final_path


def save_data_pickle(save_path, params=None, data=None, sweep_arrays=None, derived_arrays=None, overwrite=True):
    """
    Saves the simulation data using pickle-function.

    Parameters
    ----------
    save_path : str
        Path to the directory where the data will be saved. If save_path is not a directory, a new directory will be created where the file extension is stripped from the save_path and the resulting string is used as a save directory.
        Example: path/to/data.hdf5 -> path/to/data/
    params : dict, optional
        Dictionary of the simulation parameters. If not None, will be saved to load_path/params.obj.
    data : object, optional
        Data structure containing the simulation data. If not None, will be saved to load_path/data.obj.
    sweep_arrays : dict, optional
        Dictionary containing the sweep arrays. If not None, will be saved to load_path/sweep_arrays.obj.
    derived_arrays : dict, optional
        Dictionary containing the derived_arrays. If not None, will be saved to load_path/derived_arrays.obj.
    Returns
    -------
    str :
        Directory to which the data was saved.

    """
    save_dir = os.path.splitext(save_path)[0]
    if not overwrite:
        proposed_dir = save_dir
        ii = 0
        while os.path.exists(proposed_dir):
            ii = ii+1
            proposed_dir = save_dir + '_' + str(ii)
        save_dir = proposed_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if params:
        with open(os.path.join(save_dir, 'params.obj'), 'wb') as f:
            try:
                pickle.dump(params, f)
            except Exception as e:
                logging.warning(
                    'Unable to dump parameters to a file. Parameters are not saved.')
                logging.warning(e, exc_info=True)

    if sweep_arrays:
        with open(os.path.join(save_dir, 'sweep_arrays.obj'), 'wb') as f:
            try:
                pickle.dump(sweep_arrays, f)
            except Exception as e:
                logging.warning(
                    'Unable to dump sweep arrays to a file. Sweep arrays are not saved.')
                logging.warning(e, exc_info=True)

    if derived_arrays:
        with open(os.path.join(save_dir, 'derived_arrays.obj'), 'wb') as f:
            try:
                pickle.dump(derived_arrays, f)
            except Exception as e:
                logging.warning(
                    'Unable to dump derived arrays to a file. Derived arrays are not saved.')
                logging.warning(e, exc_info=True)

    if data:
        with open(os.path.join(save_dir, 'data.obj'), 'wb') as f:
            try:
                pickle.dump(data, f)
            except Exception as e:
                logging.warning(
                    'Unable to dump data to a file. Data is not saved.')
                logging.warning(e, exc_info=True)
    return save_dir


def load_pickled_data(load_path):
    """
    Loads pickled data saved to load_path directory.

    Parameters
    ----------
    load_path : str
        Path to the directory where the data is located.

    Returns
    -------
    dict :
        Dictionary containing all the data and the simulation parameters. The keys are 'data', 'params', 'sweep_arrays' and 'derived_arrays', corresponding to the object files found in the load_path directory.
    """
    out = {}
    try:
        with open(os.path.join(load_path, 'data.obj'), 'rb') as f:
            try:
                data = pickle.load(f)
                out['data'] = data
            except Exception as e:
                logging.warning('Unable to load data.obj')
    except FileNotFoundError:
        pass
    try:
        with open(os.path.join(load_path, 'params.obj'), 'rb') as f:
            try:
                params = pickle.load(f)
                out['params'] = params
            except Exception as e:
                logging.warning('Unable to load params.obj')
                pass
    except FileNotFoundError:
        pass
    try:
        with open(os.path.join(load_path, 'sweep_arrays.obj'), 'rb') as f:
            try:
                sweep_arrays = pickle.load(f)
                out['sweep_arrays'] = sweep_arrays
            except Exception as e:
                logging.warning('Unable to load sweep_arrays.obj')
                pass
    except FileNotFoundError:
        pass
    try:
        with open(os.path.join(load_path, 'derived_arrays.obj'), 'rb') as f:
            try:
                derived_arrays = pickle.load(f)
                out['derived_arrays'] = derived_arrays
            except Exception as e:
                logging.warning('Unable to load derived_arrays.obj')
                pass
    except FileNotFoundError:
        pass

    return out
