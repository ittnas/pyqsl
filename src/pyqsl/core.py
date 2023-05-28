import copy as copy
import datetime
import inspect
import json
import logging
import multiprocessing as mp
import os
import pickle
from collections.abc import Iterable
from functools import partial

import numpy as np
import tqdm
from packaging import version as pkg


def _default_save_element_fun(save_path, output, ii):
    """Saves the element using qutip qsave function

    Parameters
    ----------
    save_path : str
        The path to which the data is saved.
    output : obj
        The object to save.
    ii : int
        The serial index of the output element in the simulation.
    """
    import qutip as qutip

    qutip.qsave(output, os.path.join(save_path, "qobject_" + str(ii)))


def _default_save_parameters_function(
    full_save_path, params, sweep_arrays, derived_arrays
):
    with open(os.path.join(full_save_path, "parameters.json"), "w") as f:
        try:
            json.dump(params, f)
        except Exception as e:
            logging.warning(
                "Unable to dump parameters to a file. Parameters are not saved."
            )
            logging.warning(e, exc_info=True)
            # print('Unable to dump parameters to a file. Parameters are not saved.')
            # print('-'*60)
            # traceback.print_exc(file=sys.stdout)
            # print('-'*60)

    with open(os.path.join(full_save_path, "sweep_arrays.json"), "w") as f:
        sweep_arrays_s = {}
        for key, value in sweep_arrays.items():
            # XXX Not a proper way to save a list to json.
            sweep_arrays_s[key] = str(value)
        try:
            json.dump(sweep_arrays_s, f)
        except Exception as e:
            logging.warning(
                "Unable to dump sweep_arrays to a file. Sweep arrays are not saved."
            )
            logging.warning(e, exc_info=True)
            # print('Unable to dump sweep_arrays to a file. Sweep arrays are not saved.')
            # print('-'*60)
            # traceback.print_exc(file=sys.stdout)
            # print('-'*60)

    with open(os.path.join(full_save_path, "derived_arrays.json"), "w") as f:
        derived_arrays_s = {}
        for key, value in derived_arrays.items():
            # XXX Not a proper way to save a dict to json
            derived_arrays_s[key] = str(value)
        try:
            json.dump(derived_arrays_s, f)
        except Exception as e:
            logging.warning(
                "Unable to dump derived_arrays to a file. Derived arrays are not saved."
            )
            logging.warning(e, exc_info=True)

            # print('Unable to dump derived_arrays to a file. Derived arrays are not saved.')
            # print('-'*60)
            # traceback.print_exc(file=sys.stdout)
            # print('-'*60)


def _default_save_data_function(
    save_path, sweep_arrays, derived_arrays, output_array, save_element_function
):
    try:
        for ii, output in enumerate(output_array):
            save_element_function(save_path, output, ii)
    except Exception as e:
        logging.warning("Error in the save_element_fun. Data cannot be saved.")
        logging.warning(e, exc_info=True)


def _simulation_loop_body(
    ii,
    params,
    dims,
    sweep_arrays,
    derived_arrays,
    pre_processing_in_the_loop,
    post_processing_in_the_loop,
    simulation_task,
):
    # The main loop
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
            params_private[subkey] = derived_arrays[key][subkey][
                current_ind[derived_arrays_index]
            ]
        # print(params_private)
        derived_arrays_index = derived_arrays_index + 1

    if pre_processing_in_the_loop:
        pre_processing_in_the_loop(params_private, **params_private)

        # params_private now contains all the required information to run the simulation
    output = simulation_task(params_private, **params_private)

    if post_processing_in_the_loop:
        output = post_processing_in_the_loop(output, params_private, **params_private)
    return output


def simulation_loop(
    params,
    simulation_task,
    sweep_arrays={},
    derived_arrays={},
    pre_processing_before_loop=None,
    pre_processing_in_the_loop=None,
    post_processing_in_the_loop=None,
    parallelize=False,
    expand_data=True,
    n_cores=None,
):
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
    expand_data : bool, opt
        Flag indicating whether the first level of variables should be expanded. WARNING, DOES NOT WORK FOR DICT OUTPUTS!
    n_cores : int, opt
        Number of cores to use in parallel processing. If None, all the available cores are used. For negative numbers N_max + n_cores is used. Defaults to None.

    """
    start_time = datetime.datetime.now()
    logging.info("Simulation started at " + str(start_time))

    dims = []
    for key, value in sweep_arrays.items():
        dims.append(len(value))

    logging.debug(dims)
    if dims == []:
        dims = 1
    N_tot = np.prod(dims)
    logging.info("Sweep dimensions: " + str(dims) + ".")
    output_array = [None] * N_tot

    if pre_processing_before_loop:
        pre_processing_before_loop(params, **params)

    if parallelize:
        # Weird fix needed due to a bug somewhere in multiprocessing if running windows + jupyter
        # https://stackoverflow.com/questions/47313732/jupyter-notebook-never-finishes-processing-using-multiprocessing-python-3
        with open(f"./tmp_simulation_task.py", "w") as file:
            file.write(
                inspect.getsource(simulation_task).replace(
                    simulation_task.__name__, "task"
                )
            )
        from tmp_simulation_task import task
    else:
        task = simulation_task

    simulation_loop_body_partial = partial(
        _simulation_loop_body,
        params=params,
        dims=dims,
        sweep_arrays=sweep_arrays,
        derived_arrays=derived_arrays,
        pre_processing_in_the_loop=pre_processing_in_the_loop,
        post_processing_in_the_loop=post_processing_in_the_loop,
        simulation_task=task,
    )
    if parallelize:
        if n_cores < 0:
            n_cores = os.cpu_count() + n_cores
        os.cpu_count()
        with mp.Pool(processes=n_cores) as p:
            output_array = list(
                tqdm.tqdm(
                    p.imap(simulation_loop_body_partial, range(N_tot)),
                    total=N_tot,
                    smoothing=0,
                )
            )
    else:
        for ii in tqdm.tqdm(range(N_tot)):
            output = simulation_loop_body_partial(ii)
            output_array[ii] = output
    end_time = datetime.datetime.now()
    logging.info(
        "Simulation finished at "
        + str(end_time)
        + ". The duration of the simulation was "
        + str(end_time - start_time)
        + "."
    )

    if expand_data:
        if isinstance(output_array[0], dict):
            temporary_array = {}
            for key in output_array[0]:
                temporary_array[key] = []
            for ii in range(len(output_array)):
                for key in output_array[0]:
                    temporary_array[key].append(output_array[ii][key])
            for key in output_array[0]:
                new_shape = np.array(temporary_array[key]).shape[1:]
                if isinstance(dims, int):
                    new_dims = [dims]
                else:
                    new_dims = dims.copy()
                new_dims.extend(new_shape)
                if (
                    isinstance(dims, int) and dims == 1
                ):  # Make sure the first dimension is equal to number of dimensions in sweep arrays. Remove singleton dimension, if sweep_arrays = {}
                    temporary_array[key] = np.reshape(
                        np.array(temporary_array[key]), new_dims
                    )[0]
                else:
                    temporary_array[key] = np.reshape(
                        np.array(temporary_array[key]), new_dims
                    )
            return temporary_array
        elif isinstance(output_array[0], Iterable):
            output_array = list(zip(*output_array))
            for ii in range(len(output_array)):
                new_shape = (np.array(output_array[ii]).shape)[1:]
                new_dims = dims.copy()
                new_dims.extend(new_shape)
                output_array[ii] = np.reshape(np.array(output_array[ii]), new_dims)
            return output_array
        else:
            new_dims = dims.copy()
            new_dims.append(-1)
            # Not a great solution, adds a singleton dimension.
            output_array = np.reshape(np.array(output_array), new_dims)
            return output_array
    else:
        # No reshaping done. Fix this.
        return np.reshape(np.array(output_array), dims)


def save_data(
    save_path,
    output_array,
    params,
    sweep_arrays,
    derived_arrays,
    save_element_fun=_default_save_element_fun,
    save_parameters_function=_default_save_parameters_function,
    save_data_function=_default_save_data_function,
    use_date_directory_structure=True,
):
    """Saves the data to a directory given by save_path/data/current_date if use_date_directory_structure is True. Otherwise saves the data to save_path/. DEPRACATED.

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
        full_save_path = os.path.join(save_path, "data", str(datetime.datetime.now()))
    else:
        full_save_path = save_path
    try:
        os.makedirs(full_save_path)
    except FileExistsError:
        # path already exists, just continue
        pass
    except OSError as err:
        logging.error(
            "Error while creating the saving directory. Data cannot be saved."
        )
        logging.error(err, exc_info=True)
        # print('Error while creating the saving directory. Data cannot be saved. The error is:',sys.exc_info()[1])

    # Saving parameters
    save_parameters_function(full_save_path, params, sweep_arrays, derived_arrays)
    save_data_function(
        full_save_path, sweep_arrays, derived_arrays, output_array, save_element_fun
    )
    return full_save_path


def save_data_pickle(
    save_path,
    params=None,
    data=None,
    sweep_arrays=None,
    derived_arrays=None,
    overwrite=True,
):
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
            ii = ii + 1
            proposed_dir = save_dir + "_" + str(ii)
        save_dir = proposed_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if params:
        with open(os.path.join(save_dir, "params.obj"), "wb") as f:
            try:
                pickle.dump(params, f)
            except Exception as e:
                logging.warning(
                    "Unable to dump parameters to a file. Parameters are not saved."
                )
                logging.warning(e, exc_info=True)

    if sweep_arrays:
        with open(os.path.join(save_dir, "sweep_arrays.obj"), "wb") as f:
            try:
                pickle.dump(sweep_arrays, f)
            except Exception as e:
                logging.warning(
                    "Unable to dump sweep arrays to a file. Sweep arrays are not saved."
                )
                logging.warning(e, exc_info=True)

    if derived_arrays:
        with open(os.path.join(save_dir, "derived_arrays.obj"), "wb") as f:
            try:
                pickle.dump(derived_arrays, f)
            except Exception as e:
                logging.warning(
                    "Unable to dump derived arrays to a file. Derived arrays are not saved."
                )
                logging.warning(e, exc_info=True)

    if data:
        with open(os.path.join(save_dir, "data.obj"), "wb") as f:
            try:
                pickle.dump(data, f)
            except Exception as e:
                logging.warning("Unable to dump data to a file. Data is not saved.")
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
        with open(os.path.join(load_path, "data.obj"), "rb") as f:
            try:
                data = pickle.load(f)
                out["data"] = data
            except Exception as e:
                logging.warning("Unable to load data.obj")
    except FileNotFoundError:
        pass
    try:
        with open(os.path.join(load_path, "params.obj"), "rb") as f:
            try:
                params = pickle.load(f)
                out["params"] = params
            except Exception as e:
                logging.warning("Unable to load params.obj")
                pass
    except FileNotFoundError:
        pass
    try:
        with open(os.path.join(load_path, "sweep_arrays.obj"), "rb") as f:
            try:
                sweep_arrays = pickle.load(f)
                out["sweep_arrays"] = sweep_arrays
            except Exception as e:
                logging.warning("Unable to load sweep_arrays.obj")
                pass
    except FileNotFoundError:
        pass
    try:
        with open(os.path.join(load_path, "derived_arrays.obj"), "rb") as f:
            try:
                derived_arrays = pickle.load(f)
                out["derived_arrays"] = derived_arrays
            except Exception as e:
                logging.warning("Unable to load derived_arrays.obj")
                pass
    except FileNotFoundError:
        pass

    return out
