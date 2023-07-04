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
from typing import Any, Optional, Sequence, Union

import numpy as np
import tqdm
from packaging import version as pkg

from pyqsl.settings import Setting, Settings
from pyqsl.simulation_result import SimulationResult
import xarray as xr

logger = logging.getLogger(__name__)


def _simulation_loop_body(
    ii: int,
    settings: Settings,
    dims,
    sweeps,
    pre_processing_in_the_loop,
    post_processing_in_the_loop,
    task,
):
    # The main loop
    # Make sure that parallel threads don't simulataneously edit params. Only use params_private in the following
    settings = copy.deepcopy(settings)
    # settings_dict = settings.to_dict()
    current_ind = np.unravel_index(ii, dims)
    sweep_array_index = 0
    for key, value in sweeps.items():
        # Update all the parameters
        sweep_name: str
        try:
            sweep_name = key.name
        except AttributeError:
            sweep_name = key
        setattr(settings, sweep_name, value[current_ind[sweep_array_index]])
        sweep_array_index = sweep_array_index + 1

    if pre_processing_in_the_loop:
        pre_processing_in_the_loop(settings)

        # params_private now contains all the required information to run the simulation
    # Resolve relations
    settings_with_relations = settings.resolve_relations()
    settings_dict = settings.to_dict()
    invalid_args = _get_invalid_args(task, settings_dict)
    logger.debug('Removing invalid args ({str(invalid_args)}) from function.')
    valid_settings = {key: settings_dict[key] for key in settings_dict if key not in invalid_args}
    output = task(**valid_settings)

    if post_processing_in_the_loop:
        output = post_processing_in_the_loop(output, **settings_dict)

    output_as_dict = {'output': output, 'settings_with_relations': {key: settings_dict[key] for key in settings_with_relations}}
    return output_as_dict


def run(
    task,
    settings: Settings,
    sweeps: dict[Union[str, Setting], Sequence] = {},
    pre_processing_before_loop=None,
    pre_processing_in_the_loop=None,
    post_processing_in_the_loop=None,
    parallelize: bool = False,
    expand_data: bool = True,
    n_cores: Optional[int] = None,
    jupyter_compability_mode: bool = False,
) -> xr.Dataset:
    """
    Runs the simulation.

    Args:
        settings: settings for the run
        simulation_task :
            A function that performs the simulation. Should have form [output = simulation_task(params)], where output
            is the result of the simulation.
        sweep_arrays:
            A dictionary containing the parameters that are being swept as keys and arrays of swept parameters as values.
        derived_arrays:
            A dictionary containing dictionaries of parameters that are related to parameters in sweep_arrays
        pre_processing_before_loop:
            Function to pre-process the parameter array. Takes params dictionary as an input.
        pre_processing_in_the_loop:
            Modifies the parameter array in the loop. All the parameters that are dependant on the swept parameters should be recalculated here.
        post_processing_in_the_loop:
            Function can be used to modify the output of the simulation task. Takes params as an input.
        parallelize:
            Boolean indicating whether the computation should be parallelized.
        expand_data:
            Flag indicating whether the first level of variables should be expanded. WARNING, DOES NOT WORK FOR DICT OUTPUTS!
        n_cores:
            Number of cores to use in parallel processing. If None, all the available cores are used. For negative numbers N_max + n_cores is used. Defaults to None.
        jupyter_compability_mode:
            If running in jupyter on windows, this needs to be set to True. This is due to a weird behaviour in multiprocessing, which requires
            the task to be saved to a file.

    """
    start_time = datetime.datetime.now()
    logger.info("Simulation started at " + str(start_time))
    dims = [len(sweep_values) for sweep_values in sweeps.values()]

    N_tot = int(np.prod(dims))
    logger.info("Sweep dimensions: " + str(dims) + ".")
    output_array = [None] * N_tot

    settings = copy.deepcopy(settings)

    if pre_processing_before_loop:
        pre_processing_before_loop(settings)

    windows_and_jupyter = jupyter_compability_mode
    if parallelize and windows_and_jupyter:
        # Weird fix needed due to a bug somewhere in multiprocessing if running windows + jupyter
        # https://stackoverflow.com/questions/47313732/jupyter-notebook-never-finishes-processing-using-multiprocessing-python-3
        with open(f"./tmp_simulation_task.py", "w") as file:
            file.write(inspect.getsource(task).replace(task.__name__, "task"))
        # import sys
        # from pprint import pprint
        # pprint(sys.path)
        # This does not work with pytest
        from tmp_simulation_task import task

    simulation_loop_body_partial = partial(
        _simulation_loop_body,
        settings=settings,
        dims=dims,
        sweeps=sweeps,
        pre_processing_in_the_loop=pre_processing_in_the_loop,
        post_processing_in_the_loop=post_processing_in_the_loop,
        task=task,
    )
    if parallelize:
        if n_cores is not None:
            if n_cores < 0:
                n_cores = np.max([os.cpu_count() + n_cores, 1])
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
    logger.info(
        "Simulation finished at "
        + str(end_time)
        + ". The duration of the simulation was "
        + str(end_time - start_time)
        + "."
    )
    dataset = _create_dataset(output_array, settings, sweeps, expand_data=expand_data, dims=dims)
    simulation_result = SimulationResult(dataset)
    return simulation_result


def _create_dataset(output_array: Any, settings: Settings, sweeps: dict[Union[str, Setting], Any], expand_data, dims)->xr.Dataset:
    """
    Creates xarray dataset from simulation results.
    """
    coords = {setting.name if isinstance(setting, Setting) else setting: data for setting, data in sweeps.items()}
    dataset: xr.Dataset

    data_vars: dict[str, Any] = {}

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
        temporary_array[key] = np.reshape(
            np.array(temporary_array[key]), new_dims
        )

    expanded_output = temporary_array
    output_array = expanded_output['output']
    relation_array = expanded_output['settings_with_relations']
    relation_dict: dict[str, Any]
    if len(dims) == 0:
        output_array_reshaped = output_array[()]
        relation_dict = {f'{key}_evaluated': value for key, value in relation_array[()].items()}
    else:
        try:
            output_array_reshaped = np.reshape(np.array(output_array), dims)
        except ValueError:
            output_array_reshaped = np.reshape(np.array(output_array), dims + [-1])
        relation_dict = {}
        for key in relation_array.flat[0]:
            relation_dict[f'{key}_evaluated'] = []
        for ii in range(relation_array.size):
            for key_rd, key_ra in zip(relation_dict.keys(), relation_array.flat[0].keys()):
                relation_dict[key_rd].append(relation_array.flat[ii][key_ra])
        for key in relation_dict:
            new_shape = np.array(relation_dict[key]).shape[1:]
            new_dims = dims.copy()
            new_dims.extend(new_shape)
            relation_dict[key] = np.reshape(
                np.array(relation_dict[key]), new_dims
            )

    if expand_data:
        if isinstance(output_array.flat[0], dict):
            temporary_array = {}
            for key in output_array.flat[0]:
                temporary_array[key] = []
            for ii in range(output_array.size):
                for key in output_array.flat[0]:
                    temporary_array[key].append(output_array.flat[ii][key])
            for key in output_array.flat[0]:
                new_shape = np.array(temporary_array[key]).shape[1:]
                if isinstance(dims, int):
                    new_dims = [dims]
                else:
                    new_dims = dims.copy()
                new_dims.extend(new_shape)
                temporary_array[key] = np.reshape(
                    np.array(temporary_array[key]), new_dims
                )
            for key in temporary_array:
                data_vars[key] = (tuple(coords), temporary_array[key])
            #return temporary_array
        elif output_array.shape != tuple(dims):
            # Iterable that has been converted to np.array
            for ii in range(len(dims)):
                output_array_reshaped = np.moveaxis(output_array_reshaped, 0, -1)
            extended_coords = tuple([f'index_{ii}' for ii in range(len(output_array_reshaped.shape) - len(dims))]) + tuple(coords)
            data_vars['data'] = (extended_coords, output_array_reshaped)
        else:
            # Fallback to default behaviour without expanding
            data_vars['data'] = (tuple(coords), output_array_reshaped)
    else:
        extended_coords = tuple(coords) + tuple([f'index_{ii}' for ii in range(len(output_array_reshaped.shape) - len(dims))])
        data_vars['data'] = (tuple(extended_coords), output_array_reshaped)

    for setting_name, value in relation_dict.items():
        data_vars[setting_name] = (tuple(coords), value)

    #print(data_vars, coords)
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={'settings': settings}
    )

    return dataset


def _get_invalid_args(func, argdict):
    """
    Get set of invalid arguments for a function.
    https://stackoverflow.com/questions/196960/can-you-list-the-keyword-arguments-a-function-receives
    """
    args, varargs, varkw, _, kwonlyargs, _,  _ = inspect.getfullargspec(func)
    if varkw:
        return set()  # All accepted
    return set(argdict) - set(args)
