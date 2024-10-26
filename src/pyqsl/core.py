"""
This module contains the core functionality of pyqsl simulation loop.

pyqsl simulation is done by calling the ``run`` function in this module.
"""
import collections
import copy
import datetime
import inspect
import logging
import multiprocessing as mp
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
import psutil
import tqdm.auto as tqdm
import xarray as xr

from pyqsl.common import (
    SweepsStandardType,
    SweepsType,
    TaskOutputType,
    _get_settings_for_resolve_in_loop,
    calculate_chunksize,
    convert_sweeps_to_standard_form,
    create_numpy_array_with_fixed_dimensions,
    resolve_relations_with_sweeps,
    vstack_and_reshape,
)
from pyqsl.settings import Settings
from pyqsl.simulation_result import SimulationResult

logger = logging.getLogger(__name__)


def run(
    task: None | Callable[..., TaskOutputType],
    settings: Optional[Settings] = None,
    sweeps: Optional[SweepsType] = None,
    pre_process_before_loop: Optional[Callable[[Settings], None]] = None,
    pre_process_in_loop: Optional[Callable[[Settings], None]] = None,
    post_process_in_loop: Optional[
        Callable[[Settings, TaskOutputType], TaskOutputType]
    ] = None,
    parallelize: bool = False,
    expand_data: bool = True,
    n_cores: Optional[int] = None,
    jupyter_compatibility_mode: bool = False,
    use_shallow_copy: bool = False,
    disable_progress_bar: bool = False,
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

    The execution of the task and the evaluation of the relations can be parallelized using multi-processing
    by providing the argument ``parallelize=True``. While the parallelization works well in most cases, there
    are situations when the parallelization overhead is higher than the benefit.

    * It is not recommended to parallelize the execution if the task function contains an internal
      parallelization mechanism. For example, ``numpy.linalg.inv`` is already parallized which results in very
      inefficient execution if ``parallize`` is set to `True` for tasks requiring matrix inversion.
    * When processing large datasets, copying the data between the processes might result in a large overhead.
      Operating systems supporting ``fork()`` in principle do not create a copy of read-only data, but in
      python just accessing an object increments it's reference count, triggering the copy logic.
    * For parallelization, pyqsl uses ``multiprocessing`` library, which is not fully supported with
      interactive interpreters such as jupyter notebook. The execution might hang unexpectedly.
    * Multiprocessing with interactive interpreter and Windows OS is problematic and likely results in a crash.
      In order to avoid one of the known problems, write your task function in a separate python file instead
      of e.g a cell of a jupyter notebook. You can also try setting ``jupyter_compatibility_mode=True``.

    Args:
        task:
            A reference to the function used for simulation. The function can accept any number of inputs
            and should return either a tuple, a single number or a dictionary. Can also be a sequence of
            tasks, in which case all of them are executed in order.
        settings: Settings for the run.
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
        jupyter_compatibility_mode:
            If running in jupyter on windows, this needs to be set to True. This is due to a weird behaviour in
            multiprocessing, which requires the task to be saved to a file. When using this mode, the task
            function needs to be written in a very specific way. For example, all the imports needed
            by the function need to be done within the function definition.
        use_shallow_copy:
            If True, only a shallow copies of Settings are made. Setting to True might provide a small
            improvement in performance if Settings contains large amounts of data. However, the user
            must ensure that the task does not modify the objects in Settings during the execution.
       disable_progress_bar:
            If True, does not show the progress bar during execution.

    Returns:
        :class:`~.SimulationResult` that contains the resulting data and a copy of the settings.

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

            settings = pyqsl.Settings()
            settings.a = 1
            settings.b = 2
            result = pyqsl.run(simulation_task, settings)
            print(result.c)

        >>> 3

        .. code-block:: python

            import pyqsl
            import numpy as np

            def simulation_task(a:int , b:float) -> float:
                return {'c': a + b}

            settings = pyqsl.Settings()
            settings.a = 1
            settings.b = 2
            result = pyqsl.run(simulation_task, settings, sweeps={'a': np.linspace(0, 1, 101)})
            print(result.c.shape)

        >>> (101, )

    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    start_time = datetime.datetime.now()
    logger.info("Simulation started at %s", str(start_time))
    settings = settings or Settings()
    if use_shallow_copy:
        settings = settings.copy()
    else:
        settings = copy.deepcopy(settings)

    sweeps_std_form = {} if sweeps is None else convert_sweeps_to_standard_form(sweeps)
    dims = [len(sweep_values) for sweep_values in sweeps_std_form.values()]

    N_tot = int(np.prod(dims))  # pylint: disable=invalid-name
    logger.info("Sweep dimensions: %s.", str(dims))
    output_array: list[Optional[dict[str, Any]]] = [None] * N_tot

    if pre_process_before_loop:
        pre_process_before_loop(settings)

    windows_and_jupyter = jupyter_compatibility_mode
    if parallelize and windows_and_jupyter:
        # Weird fix needed due to a bug somewhere in multiprocessing if running windows + jupyter
        # https://stackoverflow.com/questions/47313732/jupyter-notebook-never-finishes-processing-using-multiprocessing-python-3

        if not callable(task):
            raise ValueError(
                f"When using 'jupyter_compatibility_mode' task must be Callable. Got {type(task)}."
            )
        with open("./tmp_simulation_task.py", "w", encoding="utf-8") as file:
            file.write(
                inspect.getsource(task).replace(task.__name__, "simulation_task")
            )

        # Note, this does not work within pytest

        from tmp_simulation_task import (  # pylint: disable=import-error,import-outside-toplevel
            simulation_task,
        )
    else:
        simulation_task = task

    settings.resolve_relations()
    pool: LinearPool | mp.pool.Pool
    if parallelize:
        cores = psutil.Process().cpu_affinity()
        max_nbr_cores = len(cores) if cores else 1
        used_cores = (
            max_nbr_cores
            if n_cores is None
            else np.max([max_nbr_cores + n_cores, 1])
            if n_cores < 0
            else n_cores
        )
        pool = mp.Pool(processes=used_cores)
        execution_settings = {"chunksize": calculate_chunksize(used_cores, N_tot)}
    else:
        pool = LinearPool()
        used_cores = 1
        execution_settings = {}

    with pool:
        resolved_settings_dataset = resolve_relations_with_sweeps(
            settings=settings,
            sweeps=sweeps_std_form,
            pool=pool,
            parallelize=parallelize,
            n_cores=used_cores,
            disable_progress_bar=disable_progress_bar,
        )
        setting_names_for_tasks = list(resolved_settings_dataset.data_vars)
        for sweep_name in sweeps_std_form:
            if sweep_name not in setting_names_for_tasks:
                setting_names_for_tasks.append(sweep_name)

        setting_dim_dict = {
            str(setting_name): [
                str(dim) for dim in resolved_settings_dataset[setting_name].dims
            ]
            for setting_name in resolved_settings_dataset.data_vars
        }
        setting_value_dict = {
            str(setting_name): resolved_settings_dataset[setting_name].values
            for setting_name in resolved_settings_dataset.data_vars
        }
        get_settings_task = partial(
            _get_settings_for_resolve_in_loop,
            needed_setting_names=setting_names_for_tasks,
            setting_value_dict=setting_value_dict,
            setting_dim_dict=setting_dim_dict,
            dims=dims,
            sweeps=sweeps_std_form,
            mapped_setting_names=dict(
                zip(setting_names_for_tasks, setting_names_for_tasks)
            ),
        )
        setting_value_dicts = list(map(get_settings_task, range(N_tot)))
        simulation_loop_body_partial = partial(
            _simulation_loop_body,
            settings=settings,
            sweeps=sweeps_std_form,
            pre_process_in_loop=pre_process_in_loop,
            post_processs_in_loop=post_process_in_loop,
            task=simulation_task,
            use_shallow_copy=use_shallow_copy,
        )
        output_array = list(
            tqdm.tqdm(
                pool.imap(
                    simulation_loop_body_partial,
                    setting_value_dicts,
                    **execution_settings,
                ),
                total=N_tot,
                leave=True,
                desc="Resolving tasks",
                disable=disable_progress_bar,
            )
        )

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
        expand_data=expand_data,
        dims=dims,
        resolved_settings_dataset=resolved_settings_dataset,
    )
    simulation_result = SimulationResult(dataset)
    return simulation_result


def _simulation_loop_body(
    setting_value_dict: dict[str, Any],
    settings: Settings,
    sweeps: SweepsStandardType,
    pre_process_in_loop: Callable | None,
    post_processs_in_loop: Callable | None,
    task: Callable | None | list[Callable],
    use_shallow_copy: bool,
) -> dict[str, Any]:
    """
    This function is responsible for all the actions that happen for each point in simulation.

    Args:
        setting_value_dict: Dict containing values for resolved settings
        settings:
            Settings for index ``ii`` in the simulation. At this point, the relations have not been resolved, yet.
        sweeps: Sweeps for the simulation.
        pre_process_in_loop: Callback to do processing before the task is called.
        post_process_in_loop: Callback to do post-processing after the task is called.
        task: The main task to run.
        use_shallow_copy: If True, only a shallow copy of the Settings is made.

    Returns:
        The results in a dictionary.
    """
    # pylint: disable=too-many-branches

    # Make sure that parallel threads don't simulteneously edit params.
    # Only use params_private in the following.

    original_settings = settings
    if use_shallow_copy:
        settings = original_settings.copy()
    else:
        settings = copy.deepcopy(original_settings)
    for setting_name, setting_value in setting_value_dict.items():
        # Get the setting value from previously evaluated setting

        setattr(settings, setting_name, setting_value)

    if pre_process_in_loop:
        pre_process_in_loop(settings)

    # Resolve relations

    task = task or (lambda: {})  # pylint: disable=unnecessary-lambda-assignment
    settings_dict = settings.to_dict()
    if not isinstance(task, list):
        task_list = [task]
    else:
        task_list = task

    final_result = {}
    new_settings = {}
    for current_task in task_list:
        invalid_args = _get_invalid_args(current_task, settings_dict)
        logger.debug("Removing invalid args (%s) from function.", str(invalid_args))
        valid_settings = {
            key: settings_dict[key] for key in settings_dict if key not in invalid_args
        }

        # Call the task function. If "settings" is one of the arguments,
        # substitute that with the settings object.

        if _settings_in_args(current_task):
            valid_settings["settings"] = settings
        output = current_task(**valid_settings)

        if post_processs_in_loop:
            output = post_processs_in_loop(output, settings)

        # If any setting has been changed, add as a result.
        if isinstance(output, Settings):
            new_output = {}
            for setting in output:
                name = setting.name
                if name is None:
                    continue
                if name in sweeps:
                    continue
                if setting not in original_settings:
                    new_output[name] = setting.value
                    new_settings[name] = setting
                    continue
                comparison = False
                try:
                    # Comparison for normal setting values.

                    comparison = setting.value != original_settings[name].value
                    if (
                        comparison
                    ):  # This is necessary to catch results that cannot be compared.
                        pass
                except Exception:  # pylint: disable=broad-exception-caught
                    try:
                        # Comparison for array-like setting values.
                        comparison = (
                            setting.value != original_settings[name].value
                        ).any()
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
                    if comparison:
                        new_output[name] = setting.value
            output = new_output
        if isinstance(output, dict):
            final_result.update(output)
        else:
            final_result = output

    output_as_dict = {
        "output": final_result,
        "settings_with_relations": {},  # TODO: REMOVE THIS
        "new_settings": new_settings,
    }
    return output_as_dict


def _create_dataset(
    output_array: Any,
    settings: Settings,
    sweeps: SweepsStandardType,
    expand_data,
    dims,
    resolved_settings_dataset: xr.Dataset,
) -> xr.Dataset:
    """
    Creates xarray dataset from simulation results.

    Args:
        data_coordinates: dict of data coordinate names and their additional sweeps.
        resolved_settings_dataset: Dataset containing the resolved relations for sweeps.
    """
    # pylint: disable=too-many-branches,too-many-statements,too-many-locals

    reshaped_array = np.reshape(np.array(output_array, dtype=object), dims)
    # Expand
    reshaped_array_expanded = _expand_dict_from_data(reshaped_array)
    dataset: xr.Dataset
    data_vars: dict[str, tuple[tuple[str, ...], Any, dict]] = {}
    extended_coords: dict[str, Any] = {
        sweep: create_numpy_array_with_fixed_dimensions(
            sweeps[sweep], tuple([dims[ii]])
        )
        for ii, sweep in enumerate(sweeps)
    }

    # Add all new settings created by the task as a setting in the dataset.

    new_settings_dict = reshaped_array_expanded["new_settings"].flat[0]
    for setting_name, setting in new_settings_dict.items():
        settings[setting_name] = setting

    # Add settings as variables

    if len(dims):
        _add_settings_as_variables(
            data_vars,
            resolved_settings_dataset,
        )

    if not expand_data:
        coord_names = tuple(sweeps) + tuple(
            f"index_{ii}"
            for ii in range(len(reshaped_array_expanded["output"].shape) - len(dims))
        )
        data_vars["data"] = (coord_names, reshaped_array_expanded["output"], {})
    else:
        # Check type from first element

        first_element = reshaped_array.flat[0]["output"]
        if isinstance(first_element, collections.abc.Mapping):
            output_array_expanded = _expand_dict_from_data(
                reshaped_array_expanded["output"]
            )
            for key, value in output_array_expanded.items():
                data_vars[key] = (tuple(sweeps), value, {})
        elif isinstance(first_element, (collections.abc.Sequence, np.ndarray)) and (
            not isinstance(first_element, str)
        ):
            output_array_expanded = np.array(
                _expand_sequence_from_data(reshaped_array_expanded["output"], dims)
            )
            extended_coords["index"] = np.arange(
                output_array_expanded.size // np.prod(dims)
            )
            coord_names = ("index",) + tuple(sweeps)
            coord_names = coord_names + tuple(
                f"dummy_{ii}"
                for ii in range(len(output_array_expanded.shape) - len(coord_names))
            )
            data_vars["data"] = (coord_names, output_array_expanded, {})
        else:
            data_vars["data"] = (tuple(sweeps), reshaped_array_expanded["output"], {})

    if len(dims) == 0:
        # A special check to convert arrays to zero dimensional numpy arrays to protect
        # the underlying data structure.

        for data_var, entry in data_vars.items():
            data_vars[data_var] = (
                entry[0],
                create_numpy_array_with_fixed_dimensions(entry[1], dims),
                entry[2],
            )
    _add_dimensions_to_data_var(
        data_vars,
        settings,
        extended_coords,
        setting_values=resolved_settings_dataset,
    )
    for data_var, entry in data_vars.items():
        data_vars[data_var] = (
            entry[0],
            entry[1],
            {"units": settings[data_var].unit if data_var in settings else ""},
        )
    for coord_name, entry in extended_coords.items():
        extended_coords[coord_name] = xr.DataArray(
            entry,
            dims=(coord_name,),
            attrs={
                "units": settings[coord_name].unit if coord_name in settings else ""
            },
        )

    # Remove sweeps from datavars.

    for sweep in sweeps:
        if sweep in data_vars:
            del data_vars[sweep]

    # Try convert data that has numpy object type to any natural datatype.
    # If conversion cannot be done, retain in original form.

    data_vars_converted = {}
    for data_var, entry in data_vars.items():
        try:
            dims_for_data_var = tuple(len(extended_coords[dim]) for dim in entry[0])
            data_vars_converted[data_var] = (
                entry[0],
                create_numpy_array_with_fixed_dimensions(
                    vstack_and_reshape(entry[1]), dims_for_data_var
                ),
                entry[2],
            )
        except:  # pylint: disable=bare-except
            data_vars_converted[data_var] = data_vars[data_var]
    dataset = xr.Dataset(coords=extended_coords, attrs={"settings": settings})
    for data_var, entry in data_vars_converted.items():
        try:
            data_array = xr.DataArray(dims=entry[0], data=entry[1], attrs=entry[2])
            dataset[data_var] = data_array
        except:  # pylint: disable=bare-except
            logger.warning(
                "Unable to convert data_var %s when creating dataset.", data_var
            )
            dv_in_object_form = data_vars[data_var]
            data_array = xr.DataArray(
                dims=dv_in_object_form[0],
                data=dv_in_object_form[1],
                attrs=dv_in_object_form[2],
            )
            dataset[data_var] = data_array

    dataset = dataset.pint.quantify()
    return dataset


def _add_settings_as_variables(
    data_vars: dict[str, Any],
    setting_values: xr.Dataset,
):
    """
    Adds settings with resolved relations as data variables.

    Additionally, adds dimensions of all the settings as coordinates.
    """
    for setting_name in setting_values:
        data_vars[str(setting_name)] = (
            setting_values[setting_name].dims,
            setting_values[setting_name].values,
            {},
        )


def _add_dimensions_to_data_var(
    data_vars: dict[str, tuple[tuple[str, ...], Any, dict]],
    settings: Settings,
    sweeps: SweepsStandardType,
    setting_values: xr.Dataset,
):
    """
    Adds settings with dimensions as data vars. Also adds dimensions to data_vars with names that align with settings.
    """
    for setting in settings:
        if setting.dimensions:
            for dimension in setting.dimensions:
                if dimension in data_vars:
                    logger.warning(
                        "Dimension %s of setting %s is varied, which is not allowed.",
                        dimension,
                        setting.name,
                    )
                sweeps.update(
                    dict(
                        zip(
                            setting.dimensions,
                            [
                                settings[dimension].value
                                for dimension in setting.dimensions
                            ],
                        )
                    )
                )
            if setting.name not in data_vars:
                data_vars[setting.name] = (
                    setting.dimensions,
                    vstack_and_reshape(setting_values[setting.name].values)
                    if setting.name in setting_values
                    else setting.value,
                    {},
                )
            else:
                data_vars[setting.name] = (
                    tuple(data_vars[setting.name][0]) + tuple(setting.dimensions),
                    vstack_and_reshape(data_vars[setting.name][1]),
                    {},
                )


def _make_list_unique(seq):
    """
    Makes the list unique and ordered.

    From https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order.
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
            if key in dictionary:  # pylint: disable=cell-var-from-loop
                return dictionary[key]  # pylint: disable=cell-var-from-loop
            return np.NAN

        func = np.vectorize(value_for_key, otypes=[object])
        values = func(data)
        output[key] = values
    return output


def _expand_sequence_from_data(data: np.ndarray, dims):
    total_length = 0
    if len(dims) == 0:
        result = np.zeros((len(data),), dtype="O")
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
    for ii in range(total_length):

        def value_for_index(sequence):
            try:
                if ii < len(sequence):  # pylint: disable=cell-var-from-loop
                    return sequence[ii]  # pylint: disable=cell-var-from-loop
                return np.NAN
            except TypeError:
                # Scalar
                return np.NAN

        func = np.vectorize(value_for_index, otypes=[object])
        values = func(data)
        output.append(values)
    return output


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


class LinearPool:
    """
    Implements a context-manager for executing linear tasks.

    Supports three functions, ``map``, and ``imap``, and ``imap_unordered`` which
    all call the built-in ``map`` function.
    """

    def __init__(self):
        self.map = map
        self.imap = map
        self.imap_unordered = map

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        """
        Returns False if any errors occurred, otherwise True.
        """
        if exc_type is not None:
            return False
        return True
