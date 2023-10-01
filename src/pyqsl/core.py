"""
This module contains the core functionality of pyqsl simulation loop.

pyqsl simulation is done by calling the ``run`` function in this module.
"""
import copy
import datetime
import inspect
import logging
import multiprocessing as mp
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
import psutil
import tqdm
import xarray as xr

from pyqsl.common import (
    DataCoordinatesType,
    SweepsStandardType,
    SweepsType,
    TaskOutputType,
    convert_data_coordinates_to_standard_form,
    convert_sweeps_to_standard_form,
)
from pyqsl.settings import Settings
from pyqsl.simulation_result import SimulationResult

logger = logging.getLogger(__name__)


def _simulation_loop_body(
    ii: int,
    settings: Settings,
    dims,
    sweeps: SweepsStandardType,
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
    for sweep_name, value in sweeps.items():
        # Update all the parameters
        setattr(settings, sweep_name, value[int(current_ind[sweep_array_index])])
        sweep_array_index = sweep_array_index + 1

    if pre_processing_in_the_loop:
        pre_processing_in_the_loop(settings)

        # params_private now contains all the required information to run the simulation
    # Resolve relations
    settings_with_relations = settings.resolve_relations()
    settings_dict = settings.to_dict()
    invalid_args = _get_invalid_args(task, settings_dict)
    logger.debug("Removing invalid args (%s) from function.", str(invalid_args))
    valid_settings = {
        key: settings_dict[key] for key in settings_dict if key not in invalid_args
    }
    output = task(**valid_settings)

    if post_processing_in_the_loop:
        output = post_processing_in_the_loop(output, settings)

    output_as_dict = {
        "output": output,
        "settings_with_relations": {
            key: settings_dict[key] for key in settings_with_relations
        },
    }
    return output_as_dict


def run(
    task: Callable[..., TaskOutputType],
    settings: Settings,
    sweeps: Optional[SweepsType] = None,
    pre_process_before_loop: Optional[Callable[[Settings], None]] = None,
    pre_process_in_loop: Optional[Callable[[Settings], None]] = None,
    post_process_in_loop: Optional[
        Callable[[Settings, TaskOutputType], TaskOutputType]
    ] = None,
    post_process_after_loop: Optional[
        Callable[[Settings, SweepsStandardType], DataCoordinatesType]
    ] = None,
    parallelize: bool = False,
    expand_data: bool = True,
    n_cores: Optional[int] = None,
    jupyter_compability_mode: bool = False,
) -> SimulationResult:
    """
    Runs the simulation for a given task.

    The task function defines the workload for the simulation, which is run for each combination of the
    function input arguments defined in sweeps.
    The other input arguments for the function can be provided using the settings keyword argument.

    For example, if the task function has input arguments ``a`` and ``b``, the values ``settings.a`` and
    ``settings.b`` will be used to fetch their values.
    Sweeping the value of input argument ``a`` for example from 0 to 2 can be accomplished by defining a
    sweep in ``sweeps = {'a': [0, 1, 2]}``. The values of different settings can also depend on each other
    through relations, see documentation for ``Settings`` for additional details.

    The settings can be dynamically modified using callbacks which will be applied at several different
    stages of the evaluation process. The callback ``pre_process_before_loop`` will be called before looping
    over different sweeps, and should take settings object as an input and make necessary modifications.
    ``pre_process_in_loop`` will be applied separatately inside the simulation loop. The final callback,
    ``post_process_in_loop``` will be applied after the simulation is finished, and takes the output of the
    task and the settings as input. ``post_process_after_loop`` can be used to modify sweeps, for example
    one can add an extra coordinate that corresponds to an additional dimension the simulation creates.

    The results of the simulation are returned in ``SimulationResult`` object. The structure of the object
    depends on the task function's return type and the value of ``expand_data``. If ``expand_data`` is
    False all the data will be stored under ``result.data``. Otherwise, the type of the task output
    determines how the data is stored.

    * If task function returns a dict, the values corresponding to the keys can be accessed as attributes
      of ``result``.
    * If task returns a list or tuple, the data can can be accessed as ``result.data[i]``, where ``i`` refer
      to indices in the list or tuple.
    * If task returns a single number, the data can be accessed as ``result.data``.

    Args:
        task:
            A reference to the function used for simulation. The function can accept any number of inputs
            and should return either a tuple, a single number or a dictionary.
        settings: settings for the run
        sweeps:
            A dictionary containing the parameters that are being swept as keys and arrays of swept
            parameters as values.
        pre_process_before_loop:
            Function to pre-process the parameter array.
        pre_process_in_loop:
            Modifies the parameter array in the loop. All the parameters that are dependant on the
            swept parameters should be recalculated here.
        post_process_in_loop:
            Function can be used to modify the output of the simulation task.
        parallelize:
            Boolean indicating whether the computation should be parallelized.
        expand_data:
            Flag indicating whether the first level of variables should be expanded.
        n_cores:
            Number of cores to use in parallel processing. If None, all the available cores are used (``N_max``).
            For negative numbers, ``N_max + n_cores`` is used.
        jupyter_compability_mode:
            If running in jupyter on windows, this needs to be set to True. This is due to a weird behaviour in
            multiprocessing, which requires the task to be saved to a file. When using this mode, the task
            function needs to be written in a very specific way. For example, all the imports needed
            by the function need to be done within the function definition.

    Returns:
        SimulationResult object, that contains the resulting data and a copy of the settings.

    Examples:
        .. code-block:: python

            import pyqsl

            def simulation_task(a:int , b:float) -> float:
                return a + b

            settings = pyqsl.Settings()
            settings.a = 1
            settings.b = 2
            result = pyqsl.run(simulation_task, settings)
            print(result.data)

        >>> 3

        .. code-block:: python

            import pyqsl

            def simulation_task(a:int , b:float) -> float:
                return {'c': a + b}

            result = pyqsl.run(simulation_task, settings)
            print(result.c)

        >>> 3

        .. code-block:: python

            import pyqsl
            import numpy as np

            def simulation_task(a:int , b:float) -> float:
                return {'c': a + b}

            result = pyqsl.run(simulation_task, settings, sweeps={'a': np.linspace(0, 1, 101)})
            print(result.c.shape)

        >>> (101, )

    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    start_time = datetime.datetime.now()
    logger.info("Simulation started at %s", str(start_time))
    sweeps_std_form = {} if sweeps is None else convert_sweeps_to_standard_form(sweeps)
    dims = [len(sweep_values) for sweep_values in sweeps_std_form.values()]

    N_tot = int(np.prod(dims))  # pylint: disable=invalid-name
    logger.info("Sweep dimensions: %s.", str(dims))
    output_array = [None] * N_tot

    settings = copy.deepcopy(settings)

    if pre_process_before_loop:
        pre_process_before_loop(settings)

    windows_and_jupyter = jupyter_compability_mode
    if parallelize and windows_and_jupyter:
        # Weird fix needed due to a bug somewhere in multiprocessing if running windows + jupyter
        # https://stackoverflow.com/questions/47313732/jupyter-notebook-never-finishes-processing-using-multiprocessing-python-3
        with open("./tmp_simulation_task.py", "w", encoding="utf-8") as file:
            file.write(
                inspect.getsource(task).replace(task.__name__, "simulation_task")
            )
        # Note, This does not work within pytest
        from tmp_simulation_task import (  # pylint: disable=import-error,import-outside-toplevel
            simulation_task,
        )
    else:
        simulation_task = task

    simulation_loop_body_partial = partial(
        _simulation_loop_body,
        settings=settings,
        dims=dims,
        sweeps=sweeps_std_form,
        pre_processing_in_the_loop=pre_process_in_loop,
        post_processing_in_the_loop=post_process_in_loop,
        task=simulation_task,
    )
    if parallelize:
        if n_cores is not None:
            if n_cores < 0:
                max_nbr_cores = len(psutil.Process().cpu_affinity())
                n_cores = np.max([max_nbr_cores + n_cores, 1])
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

    new_coords: DataCoordinatesType = {}
    if post_process_after_loop:
        new_coords.update(post_process_after_loop(settings, sweeps_std_form))
    end_time = datetime.datetime.now()
    logger.info(
        "Simulation finished at %s. The duration of the simulation was %s.",
        str(end_time),
        str(end_time - start_time),
    )
    dataset = _create_dataset(
        output_array,
        settings,
        sweeps_std_form,
        data_coordinates=new_coords,
        expand_data=expand_data,
        dims=dims,
    )
    simulation_result = SimulationResult(dataset)
    return simulation_result


def _create_dataset(
    output_array: Any,
    settings: Settings,
    sweeps: SweepsStandardType,
    data_coordinates: DataCoordinatesType,
    expand_data,
    dims,
) -> xr.Dataset:
    """
    Creates xarray dataset from simulation results.

    Args:
        data_coordinates: dict of data coordinate names and their additional sweeps.
    """
    # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    coords = convert_sweeps_to_standard_form(sweeps)
    data_coordinates_std_form = convert_data_coordinates_to_standard_form(
        data_coordinates
    )
    dataset: xr.Dataset

    data_vars: dict[str, Any] = {}

    temporary_array: dict[str, Any] = {}
    for key in output_array[0]:
        temporary_array[key] = []
    for ii, value in enumerate(output_array):
        for key in output_array[0]:
            temporary_array[key].append(value[key])
    for key in output_array[0]:
        data_as_array = np.array(temporary_array[key])
        new_shape = data_as_array.shape[1:]
        if isinstance(dims, int):
            new_dims = [dims]
        else:
            new_dims = dims.copy()
        new_dims.extend(new_shape)
        temporary_array[key] = np.reshape(data_as_array, new_dims)

    expanded_output = temporary_array
    output_array = expanded_output["output"]
    relation_array = expanded_output["settings_with_relations"]
    relation_dict: dict[str, Any]
    if len(dims) == 0:
        output_array_reshaped = output_array[()]
        relation_dict = {
            f"{key}_evaluated": value for key, value in relation_array[()].items()
        }
    else:
        try:
            output_array_reshaped = np.reshape(np.array(output_array), dims)
        except ValueError:
            output_array_reshaped = np.reshape(np.array(output_array), dims + [-1])
        relation_dict = {}
        for key in relation_array.flat[0]:
            relation_dict[f"{key}_evaluated"] = []
        for ii in range(relation_array.size):
            for key_rd, key_ra in zip(
                relation_dict.keys(), relation_array.flat[0].keys()
            ):
                relation_dict[key_rd].append(relation_array.flat[ii][key_ra])
        for key, value in relation_dict.items():
            new_shape = np.array(value).shape[1:]
            new_dims = dims.copy()
            new_dims.extend(new_shape)
            relation_dict[key] = np.reshape(np.array(value), new_dims)

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
            for key, value in temporary_array.items():
                data_vars[key] = (tuple(coords), value)
            # return temporary_array
        elif output_array.shape != tuple(dims):
            # Iterable that has been converted to np.array
            for ii in range(len(dims)):
                output_array_reshaped = np.moveaxis(output_array_reshaped, 0, -1)
            extended_coords = tuple(
                f"index_{ii}"
                for ii in range(len(output_array_reshaped.shape) - len(dims))
            ) + tuple(coords)
            data_vars["data"] = (extended_coords, output_array_reshaped)
        else:
            # Fallback to default behaviour without expanding
            data_vars["data"] = (tuple(coords), output_array_reshaped)
    else:
        extended_coords = tuple(coords) + tuple(
            f"index_{ii}" for ii in range(len(output_array_reshaped.shape) - len(dims))
        )
        data_vars["data"] = (tuple(extended_coords), output_array_reshaped)
    for setting_name, value in relation_dict.items():
        data_vars[setting_name] = (tuple(coords), value)

    for data_var in data_vars:  # pylint: disable=consider-using-dict-items
        if data_var in data_coordinates_std_form:
            new_mapping = (
                data_vars[data_var][0]
                + tuple(data_coordinates_std_form[data_var].keys()),
                data_vars[data_var][1],
            )
            data_vars[data_var] = new_mapping
    dataset = xr.Dataset(
        data_vars=data_vars, coords=coords, attrs={"settings": settings}
    )

    return dataset


def _get_invalid_args(func, argdict):
    """
    Get set of invalid arguments for a function.
    Adapted from https://stackoverflow.com/questions/196960/can-you-list-the-keyword-arguments-a-function-receives.
    """
    args, _, varkw, _, _, _, _ = inspect.getfullargspec(func)
    if varkw:
        return set()  # All accepted
    return set(argdict) - set(args)
