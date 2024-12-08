"""
Definitions used by more than one other module.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Sequence, Set

import networkx as nx
import numpy as np
import tqdm.auto as tqdm
import xarray as xr

from pyqsl.settings import Setting, Settings

SweepsType = dict[str | Setting, Sequence]
TaskOutputType = dict[str, Any] | tuple | list | Any
SweepsStandardType = dict[str, Sequence]
DataCoordinatesType = dict[Setting | str, SweepsType]
DataCoordinatesStandardType = dict[str, SweepsStandardType]
logger = logging.getLogger(__name__)


def convert_sweeps_to_standard_form(
    sweeps: SweepsType | SweepsStandardType,
) -> SweepsStandardType:
    """
    Converts sweeps from form that can contain settings to standard form that only has settings names.

    Additionally, ensures that conversion to numpy array does not add extra dimensions. This is required
    for using the sweep as a coordinate for xarray.

    Args:
        sweeps: Sweeps dictionary which keys can either be setting or setting name.

    Returns:
        A soft copy of sweeps in new format.

    Raises:
        ValueError if any name for a sweep is None.
    """
    names = [key.name if isinstance(key, Setting) else key for key in sweeps.keys()]
    if any(name is None for name in names):
        raise ValueError("Name for a sweep parameter must not be None.")

    # Tell mypy that there are no Nones
    names_without_none = [name for name in names if name is not None]
    new_sweep_values: list[Any] = []

    for sweep_values in sweeps.values():
        ii = 0
        for _ in sweep_values:
            ii += 1
        correct_shape = True
        try:
            array_size = np.array(sweep_values).size
            if array_size != ii:
                correct_shape = False
        except ValueError:
            correct_shape = False
        if not correct_shape:
            cast_shape = np.zeros((ii,), dtype="O")
            for jj, value in enumerate(sweep_values):
                cast_shape[jj] = value
            new_sweep_values.append(cast_shape)
        else:
            new_sweep_values.append(sweep_values)
    return dict(zip(names_without_none, new_sweep_values))


def convert_data_coordinates_to_standard_form(
    data_coordinates: DataCoordinatesType | DataCoordinatesStandardType,
) -> DataCoordinatesStandardType:
    """
    Converts data coordinates from verbose type to standard type

    Args:
        data_coordinates: Mapping from data coordinates to their additional sweeps.

    Returns:
        A soft copy of data_coordinates in new format.

    Raises:
        ValueError if name for data coordinate is None.
    """
    new_data_coordinates = {}
    for key, value in data_coordinates.items():
        new_key = key.name if isinstance(key, Setting) else key
        if new_key is None:
            raise ValueError("Name for a data coordinate must not be None.")
        new_value = convert_sweeps_to_standard_form(value)
        new_data_coordinates[new_key] = new_value
    return new_data_coordinates


def vstack_and_reshape(array: np.ndarray) -> np.ndarray:
    """
    Uses `np.vstack` to add one more dimension from object array to the main array.

    When some of the subarrays are considered individual objects, ``np.array`` cannot
    be used to reshape the array.

    Args:
        array: Array to be stacked.
    """
    dims_first_layer = array.shape
    dims_second_layer = np.array(array.flat[0]).shape
    stacked = np.vstack(array.flatten())  # type: ignore[call-overload]
    return np.reshape(stacked, dims_first_layer + dims_second_layer)


def resolve_relations_with_sweeps(
    settings: Settings,
    sweeps: SweepsStandardType,
    n_cores,
    pool,
    parallelize: bool = True,
    disable_progress_bar=False,
) -> xr.Dataset:
    """
    Resolves relations when some of them are swept.

    Attempts to resolve relations so that minimum number of evaluations are done. Instead of
    evaluating all the relations for each point in the sweep space, relations are only evaluated
    when their dependant parameters are swept.

    No checks for cyclic relations is done. They are assumed to be detected when the relations for
    settings were resolved for the first time.

    Args:
        settings: Settings object with all the relations resolved.
        sweeps: Sweeps for which the settings are re-evalauted.
        n_cores: Number of cores to use when parallelizing task.
        pool: Worker pool used for executing the task.
        parallelize: If True, settings are resolved in parellel.
        disable_progress_bar: If True, do not show progres bar.

    Returns:
        Dataset which datavars are the setting names, coordinates are the sweeps
        and values are the evaluated setting values.
    """
    dataset = xr.Dataset(
        coords={
            sweep_name: create_numpy_array_with_fixed_dimensions(
                sweep_value, tuple([len(sweep_value)])
            )
            for sweep_name, sweep_value in sweeps.items()
        }
    )
    relation_graph = settings.get_relation_hierarchy()
    many_to_many_relation_map = settings.get_many_to_many_relation_map()

    evaluated_many_to_many_relations: Set[Any] = set()
    nodes = list(nx.topological_sort(relation_graph))
    for node in (
        pbar := tqdm.tqdm(nodes, total=len(nodes), disable=disable_progress_bar)
    ):
        setting = settings[node]
        pbar.set_description(f"Resolving relations for {setting.name}")
        sweeps_for_node = []
        needed_settings_for_node = settings.get_needed_settings(setting, relation_graph)
        for sweep in sweeps:
            if sweep in needed_settings_for_node:
                sweeps_for_node.append(sweep)
        if not sweeps_for_node:
            continue
        dims = [len(sweeps[sweep]) for sweep in sweeps_for_node]
        n_tot = int(np.prod(dims))
        execution_settings = {}
        if parallelize:
            execution_settings["chunksize"] = calculate_chunksize(n_cores, n_tot)

        if setting.has_active_relation() and not isinstance(
            setting.relation, EvaluatedManyToManyRelation
        ):
            mapped_setting_names = setting.relation.get_mapped_setting_names()
            _evaluate_relation_with_sweeps(
                setting,
                settings,
                sweeps_for_node,
                needed_settings_for_node,
                dims,
                n_tot,
                sweeps,
                mapped_setting_names,
                dataset,
                pool,
            )
        elif setting.name in many_to_many_relation_map:
            _evaluate_many_to_many_relation_with_sweeps(
                setting,
                settings,
                many_to_many_relation_map,
                evaluated_many_to_many_relations,
                sweeps_for_node,
                needed_settings_for_node,
                dims,
                n_tot,
                sweeps,
                dataset=dataset,
                pool=pool,
            )

    return dataset


def _evaluate_many_to_many_relation_with_sweeps(
    setting,
    settings,
    many_to_many_relation_map,
    evaluated_many_to_many_relations,
    sweeps_for_node,
    needed_settings_for_node,
    dims,
    n_tot,
    sweeps,
    dataset,
    pool,
):
    """
    Evalautes many-to-many relations for the sweeps for the setting.

    The execution is done by calling ``pool.imap``.

    """
    #  pylint: disable=too-many-arguments
    relation = many_to_many_relation_map[setting.name]
    if relation in evaluated_many_to_many_relations:
        return
    mapped_setting_names = relation.get_mapped_setting_names()
    setting_dim_dict = {
        str(setting_name): [str(dim) for dim in dataset[setting_name].dims]
        for setting_name in dataset.data_vars
    }
    setting_value_dict = {
        str(setting_name): dataset[setting_name].values
        for setting_name in dataset.data_vars
    }
    get_settings_task = partial(
        _get_settings_for_resolve_in_loop,
        needed_setting_names=needed_settings_for_node,
        setting_value_dict=setting_value_dict,
        setting_dim_dict=setting_dim_dict,
        dims=dims,
        sweeps=sweeps,
        mapped_setting_names=mapped_setting_names,
    )
    setting_value_dicts = list(map(get_settings_task, range(n_tot)))
    evaluate_many_to_many_relation_in_loop_task = partial(
        _evaluate_relation_in_loop,
        relation=relation,
        settings=settings,
    )
    result = list(
        pool.map(evaluate_many_to_many_relation_in_loop_task, setting_value_dicts)
    )
    for (
        output_setting_name,
        function_argument_name_or_index,
    ) in relation.output_parameters.items():
        # Split the data here.

        values = [value[function_argument_name_or_index] for value in result]
        value_array = np.reshape(
            create_numpy_array_with_fixed_dimensions(
                np.array(values, dtype=object), tuple(dims)
            ),
            dims,
        )
        dataset[output_setting_name] = (tuple(sweeps_for_node), value_array)
    evaluated_many_to_many_relations.add(relation)


def _evaluate_relation_with_sweeps(
    setting,
    settings,
    sweeps_for_node,
    needed_settings_for_node,
    dims,
    n_tot,
    sweeps,
    mapped_setting_names,
    dataset,
    pool,
):
    """
    Evaluates relations for a given setting and adds the result to the dataset.

    The evaluation is done in two steps. First, all the setting values needed for
    evaluating the relation are fetched from the `setting_value_dataset` sequentially.
    The fetched setting values are then used to evaluate the relations using the pool,
    which can be either parallel or sequential.
    """
    setting_dim_dict = {
        str(setting_name): [str(dim) for dim in dataset[setting_name].dims]
        for setting_name in dataset.data_vars
    }
    setting_value_dict = {
        str(setting_name): dataset[setting_name].values
        for setting_name in dataset.data_vars
    }
    get_settings_task = partial(
        _get_settings_for_resolve_in_loop,
        needed_setting_names=needed_settings_for_node,
        setting_value_dict=setting_value_dict,
        setting_dim_dict=setting_dim_dict,
        dims=dims,
        sweeps=sweeps,
        mapped_setting_names=mapped_setting_names,
    )
    setting_value_dicts = list(map(get_settings_task, range(n_tot)))
    evaluate_relation_in_loop_task = partial(
        _evaluate_relation_in_loop,
        relation=setting.relation,
        settings=settings,
    )
    value_array = np.reshape(
        create_numpy_array_with_fixed_dimensions(
            np.array(
                list(
                    pool.map(evaluate_relation_in_loop_task, setting_value_dicts),
                ),
                dtype=object,
            ),
            tuple(dims),
        ),
        dims,
    )
    dataset[setting.name] = (tuple(sweeps_for_node), value_array)


def _evaluate_relation_in_loop(
    setting_dict: dict[str, Any], relation, settings: Settings
):
    parameter_dict = {}
    for parameter_name, setting_name in relation.parameters.items():
        if isinstance(setting_name, Relation):
            parameter_dict[parameter_name] = _evaluate_relation_in_loop(
                setting_dict, setting_name, settings
            )
        else:
            parameter_dict[parameter_name] = (
                setting_dict[setting_name]
                if setting_name in setting_dict
                else settings[setting_name].value
            )

    # parameter_dict = {
    #     parameter_name: (
    #         setting_dict[setting_name]
    #         if setting_name in setting_dict
    #         else settings[setting_name].value
    #     )
    #     for parameter_name, setting_name in relation.parameters.items()
    # }
    return relation.evaluate(**parameter_dict)


def _get_settings_for_resolve_in_loop(
    ii: int,
    needed_setting_names: list[str],
    setting_value_dict: dict[str, Any],
    setting_dim_dict: dict[str, list[str]],
    dims: list[int],
    sweeps: SweepsStandardType,
    mapped_setting_names: dict[str, str],
) -> dict[str, Any]:
    """
    Creates a dictionary of that maps the setting names to setting values for `needed_setting_names`.

    The setting values are fetched from the `setting_value_dataset`. The index `ii` and the sweep
    dimensions are used to find out which value to fetch from the dataset.

    Args:
        ii: Index of the current sweep for dimensions of the sweeps needed by the target relation.
        needed_setting_names: A list of setting names that are needed for evaluating the given relation.
        setting_value_dataset: Dataset that contains all the values for already evalauted settings.
        dims: Dimensions for sweeps for the given relation.
        sweeps: All the sweeps.
        mapped_setting_names: Mapped setting names for the given relation.
    Returns:
        A new dataset which contains the previously evaluated Setting values.

    """
    current_ind = np.unravel_index(ii, dims)
    current_sweeps = [
        setting_name for setting_name in sweeps if setting_name in needed_setting_names
    ]
    setting_values = {}
    for setting_name in mapped_setting_names:
        if setting_name in sweeps:
            # Get the setting value from the sweep

            sweep_ind = int(current_ind[current_sweeps.index(setting_name)])
            setting_values[setting_name] = sweeps[setting_name][sweep_ind]
        if setting_name in setting_value_dict:
            # Get the setting value from previously evaluated setting

            setting_name_dims = setting_dim_dict[setting_name]
            list_descriptor = tuple(
                current_ind[ii]
                for ii, sweep_name in enumerate(current_sweeps)
                if sweep_name in setting_name_dims
            )
            setting_value = setting_value_dict[setting_name][list_descriptor]
            if isinstance(setting_value, np.ndarray):
                # Convert numpy arrays that have been cast to objects back to their
                # standard type.
                setting_value = vstack_and_reshape(setting_value)
            setting_values[setting_name] = setting_value
    return setting_values


def create_numpy_array_with_fixed_dimensions(data: Any, dims: tuple[int, ...]) -> Any:
    """
    Creates a numpy array from data with dimensions given by dims.

    If directly converting the array to numpy array would result in an array of different
    shape, an object array is created instead.
    """
    conversion_passed = True
    try:
        data_array = np.array(data)
        if data_array.shape != dims:
            conversion_passed = False
    except ValueError:
        # Conversion not possible
        data_array = np.array(data, dtype=object)
        conversion_passed = False
    if not conversion_passed:
        data_array = np.zeros(dims, dtype=object)
        if dims:
            for ii in range(np.prod(dims)):
                data_array.flat[ii] = data[ii]  # [index]
        else:
            data_array[()] = data
    return data_array


def calculate_chunksize(n_cores: int, n_points: int) -> int:
    """
    Calculates the optimum chunksize for parallelization.

    Args:
        n_cores: Number of available cores.
        n_points: Number of data points to iterate over.
    """
    if n_points > 1e5:
        factor = 1.0
    elif n_points > 1e4:
        factor = 2.0
    elif n_points > 1e3:
        factor = 5.0
    else:
        factor = 10.0
    return max(int(n_points / factor) // n_cores, 1)


from pyqsl.many_to_many_relation import (  # pylint: disable=wrong-import-position
    EvaluatedManyToManyRelation,
)
from pyqsl.relation import Relation  # pylint: disable=wrong-import-position
