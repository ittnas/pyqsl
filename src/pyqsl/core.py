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
import collections

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
    pre_process_in_loop,
    post_processs_in_loop,
    task,
) -> dict[str, Any]:
    # The main loop
    # Make sure that parallel threads don't simulataneously edit params. Only use params_private in the following
    original_settings = settings
    settings = copy.deepcopy(original_settings)
    # settings_dict = settings.to_dict()
    current_ind = np.unravel_index(ii, dims)
    sweep_array_index = 0
    for sweep_name, value in sweeps.items():
        # Update all the parameters
        setattr(settings, sweep_name, value[int(current_ind[sweep_array_index])])
        sweep_array_index = sweep_array_index + 1

    if pre_process_in_loop:
        pre_process_in_loop(settings)

        # params_private now contains all the required information to run the simulation
    # Resolve relations
    settings_with_relations = settings.resolve_relations()
    settings_dict = settings.to_dict()
    invalid_args = _get_invalid_args(task, settings_dict)
    logger.debug("Removing invalid args (%s) from function.", str(invalid_args))
    valid_settings = {
        key: settings_dict[key] for key in settings_dict if key not in invalid_args
    }
    if _settings_in_args(task):
        valid_settings["settings"] = settings
    output = task(**valid_settings)

    if post_processs_in_loop:
        output = post_processs_in_loop(output, settings)

    # If any setting has been changed, add as a result.
    if isinstance(output, Settings):
        new_output = {}
        for setting in output:
            name = setting.name
            if name is None:
                continue
            if setting not in original_settings:
                new_output[name] = setting.value
                continue
            comparison = False
            try:
                comparison = setting.value != original_settings[name].value
                if comparison:
                    pass
            except Exception:
                try:
                    comparison = (setting.value != original_settings[name].value).any()
                except Exception:
                    pass

            if comparison:
                new_output[name] = setting.value
        output = new_output

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
    ``settings.b`` will be used to fetch their values. Alternatively, if task-function has argument
    called ``settings``, the values of all the settings can be accessed through that variable.

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
        pre_process_in_loop=pre_process_in_loop,
        post_processs_in_loop=post_process_in_loop,
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
    logger.debug(coords)
    data_coordinates_std_form = convert_data_coordinates_to_standard_form(
        data_coordinates
    )
    # output_array:
    # [prod(Dsweeps), dict['output': dict[Doutput], 'settings_with_relations': 'dict']
    
    # reshaped_array:
    # [Dsweeps, dict['output': dict[Doutput], 'settings_with_relations': 'dict']
    reshaped_array = np.reshape(np.array(output_array, dtype=object), dims)
    # Expand
    reshaped_array_expanded = _expand_dict_from_data(reshaped_array)

    dataset: xr.Dataset
    data_vars: dict[str, tuple[tuple[str, ...], Any]] = {}
    extended_coords = copy.copy(coords)
    # Add settings as variables
    if len(dims):
        _add_settings_as_variables(data_vars, settings, extended_coords, reshaped_array_expanded['settings_with_relations'], dims)

    if not expand_data:
        coord_names = tuple(coords) + tuple(
            f"index_{ii}" for ii in range(len(reshaped_array_expanded['output'].shape) - len(dims))
        )
        data_vars["data"] = (coord_names, reshaped_array_expanded['output'])
    else:
        # Check type from first element
        logger.debug(f'reshaped_array.flat {reshaped_array.flat[0]}')
        first_element = reshaped_array.flat[0]['output']
        if isinstance(first_element, collections.abc.Mapping):
            output_array_expanded = _expand_dict_from_data(reshaped_array_expanded['output'])
            for key, value in output_array_expanded.items():
                data_vars[key] = (tuple(coords), value)
        elif isinstance(first_element, (collections.abc.Sequence, np.ndarray)) and (not isinstance(first_element, str)):
            output_array_expanded = np.array(_expand_sequence_from_data(reshaped_array_expanded['output'], dims))
            logger.debug(f'{output_array_expanded}, {output_array_expanded.shape}')
            extended_coords['index'] = np.arange(output_array_expanded.size//np.prod(dims))
            coord_names = ('index',) + tuple(coords)
            coord_names = coord_names + tuple(f"dummy_{ii}" for ii in range(len(output_array_expanded.shape) - len(coord_names)))
            logger.debug(f'{coord_names}, {len(output_array_expanded.shape)}, {len(coord_names)}')
            data_vars["data"] = (coord_names, output_array_expanded)
        else:
            data_vars["data"] = (tuple(coords), reshaped_array_expanded['output'])
    _add_dimensions_to_data_var(data_vars, settings, extended_coords, reshaped_array_expanded['settings_with_relations'], dims)
    for data_var, entry in data_vars.items():
        data_vars[data_var] = entry[0], entry[1], {'units': settings[data_var].unit if data_var in settings else ''}
    for coord_name, entry in extended_coords.items():
        extended_coords[coord_name] = xr.DataArray(entry, dims=(coord_name,), attrs={'units': settings[coord_name].unit if coord_name in settings else ''})
    dataset = xr.Dataset(
        data_vars=data_vars, coords=extended_coords, attrs={"settings": settings}
    )
    dataset = dataset.pint.quantify()

    return dataset


def _add_settings_as_variables(data_vars: dict[str, tuple[tuple[str, ...], Any]], settings: Settings, sweeps: SweepsStandardType, setting_values, dims):
    """
    Adds settings with resolved relations as data variables.

    Additionally, adds dimensions of all the settings as coordinates.
    """
    hierarchy = settings.get_relation_hierarchy()

    # Map from setting names to setting names that define their dimensionality.
    setting_dimension_map = {}
    for setting_name in sweeps:
        setting = settings[setting_name]
        dependent_names = settings.get_dependent_setting_names(setting, hierarchy)
        for dependent_name in dependent_names:
            if dependent_name not in setting_dimension_map:
                setting_dimension_map[dependent_name] = set()
            setting_dimension_map[dependent_name].add(setting_name)
    setting_values = _expand_dict_from_data(setting_values)
    for dependent_name, sweep_names in setting_dimension_map.items():
        new_slice = [0]*len(dims)
        name_indices = []
        for sweep_name in sweep_names:
            name_index = list(sweeps.keys()).index(sweep_name)
            name_indices.append(name_index)
            new_slice[name_index] = slice(None)
        sweep_names_in_order = [list(sweeps.keys())[index] for index in sorted(name_indices)]
        logger.debug(f'{dependent_name}: {new_slice}')
        sliced_values = setting_values[dependent_name][tuple(new_slice)]
        logger.debug(f'{dependent_name}: {sliced_values}')
        logger.debug(f'{dependent_name}: {sweep_names}')
        data_vars[dependent_name] = (sweep_names_in_order, sliced_values)


def _add_dimensions_to_data_var(data_vars: dict[str, tuple[tuple[str, ...], Any]], settings: Settings, sweeps: SweepsStandardType, setting_values, dims):
    """
    Adds settings with dimensions as data vars. Also adds dimensions to data_vars with names that align with settings.
    """
    # Add setting dimensions
    for setting in settings:
        if setting.dimensions:
            for dimension in setting.dimensions:
                if dimension in data_vars:
                    logger.warning('Dimension %s of setting %s is varied, which is not allowed. Skipping dimension.'.format(dimension, setting.name))
                sweeps.update(dict(zip(setting.dimensions, [settings[dimension].value for dimension in setting.dimensions])))
            if setting.name not in data_vars:
                data_vars[setting.name] = (setting.dimensions, setting.value)
            else:
                data_vars[setting.name] = (tuple(data_vars[setting.name][0]) + tuple(setting.dimensions), _vstack_and_reshape(data_vars[setting.name][1]))
    logger.debug(data_vars)


def _make_list_unique(seq):
    """
    Makes the list unique and ordered.

    From https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def _expand_dict_from_data(data: np.ndarray):
    all_keys = []
    for individual in data.flat:
        all_keys.extend(list(individual.keys()))
    keys = _make_list_unique(all_keys)
    output = {}
    for key in keys:
        def value_for_key(dictionary):
            if key in dictionary:
                return dictionary[key]
            else:
                return np.NAN
        func = np.vectorize(value_for_key, otypes=[object])
        values = func(data)
        output[key] = values
    return output


def _expand_sequence_from_data(data: np.ndarray, dims):
    total_length = 0
    if len(dims) == 0:
        result = np.zeros((len(data),), dtype='O')
        for ii, element in enumerate(data):
            result[ii] = element
        return result
    for individual in data.flat:
        try:
            total_length = np.max([len(individual), total_length])
        except TypeError:
            # Scalar
            pass
    output = []
    logger.debug(total_length)
    for ii in range(total_length):
        def value_for_index(sequence):
            try:
                if ii < len(sequence):
                    return sequence[ii]
                else:
                    return np.NAN
            except TypeError:
                # Scalar
                return np.NAN
        func = np.vectorize(value_for_index, otypes=[object])
        values = func(data)
        logger.debug(values)
        output.append(values)
    return output


def _vstack_and_reshape(array):
    """
    Uses `np.vstack` to add one more dimension from object array to the main array.

    When some of the subarrays are considered individual objects, ``np.array`` cannot
    be used to reshape the array.

    Args:
        array: Array to be stacked.
    """
    logger.debug(array.flatten().shape)
    logger.debug(np.vstack(array.flatten()).shape)
    dims_first_layer = array.shape
    dims_second_layer = np.array(array.flat[0]).shape
    return np.reshape(np.vstack(array.flatten()), dims_first_layer + dims_second_layer)


def _get_invalid_args(func, argdict):
    """
    Get set of invalid arguments for a function.
    Adapted from https://stackoverflow.com/questions/196960/can-you-list-the-keyword-arguments-a-function-receives.
    """
    args, _, varkw, _, _, _, _ = inspect.getfullargspec(func)
    if varkw:
        return set()  # All accepted
    return set(argdict) - set(args)


def _settings_in_args(func) -> bool:
    """
    Checks if settings is in function arguments.
    """
    args, _, _, _, _, _, _ = inspect.getfullargspec(func)
    return "settings" in args
