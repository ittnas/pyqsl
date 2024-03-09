"""
Definitions used by more than one other module.
"""
from __future__ import annotations

import copy
import logging
import multiprocessing as mp
import types
from functools import partial
from typing import TYPE_CHECKING, Any, Sequence

import networkx as nx
import numpy as np
import numpy.typing as npt
import psutil
import tqdm.auto as tqdm
import xarray as xr
from pandas._libs.tslibs import conversion

from pyqsl.many_to_many_relation import EvaluatedManyToManyRelation, ManyToManyRelation
from pyqsl.settings import Setting, Settings
from multiprocessing.managers import BaseManager

SweepsType = dict[str | Setting, Sequence]
TaskOutputType = dict[str, Any] | tuple | list | Any
SweepsStandardType = dict[str, Sequence]
DataCoordinatesType = dict[Setting | str, SweepsType]
DataCoordinatesStandardType = dict[str, SweepsStandardType]
logger = logging.getLogger(__name__)

# if TYPE_CHECKING:
#     from pyqsl.many_to_many_relation import ManyToManyRelation, EvaluatedManyToManyRelation


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
    settings: Settings, sweeps: SweepsStandardType, n_cores, pool, parallelize=True
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

    Returns:
        Dataset which datavars are the setting names, coordinates are the sweeps
        and values are the evaluated setting values.
    """
    dataset = xr.Dataset(
        coords={
            sweep_name: create_numpy_array_with_fixed_dimensions(
                sweep_value, [len(sweep_value)]
            )
            for sweep_name, sweep_value in sweeps.items()
        }
    )
    BaseManager.register('DataSetWrapper', DataSetWrapper)
    manager = BaseManager()
    manager.start()
    dataset_wrapper = manager.DataSetWrapper(dataset)
    settings = settings.copy()
    relation_graph = settings.get_relation_hierarchy()
    many_to_many_relation_map = settings.get_many_to_many_relation_map()

    evaluated_many_to_many_relations = set()
    nodes = list(nx.topological_sort(relation_graph))
    for node in (pbar := tqdm.tqdm(nodes, total=len(nodes))):
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
            logger.debug(mapped_setting_names)
            _evaluate_relation_with_sweeps(
                setting,
                settings,
                sweeps_for_node,
                needed_settings_for_node,
                dims,
                n_tot,
                sweeps,
                mapped_setting_names,
                dataset_wrapper,
                pool,
                execution_settings=execution_settings,
            )
        elif setting.name in many_to_many_relation_map:
            # In many-to-many relations, figure out why mapped_setting
            # names is not enough to set.

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
                dataset,
                pool,
                execution_settings=execution_settings,
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
    execution_settings,
):
    """
    Evalautes many-to-many relations for the sweeps for the setting.

    The execution is done by calling ``pool.imap``.

    """
    relation = many_to_many_relation_map[setting.name]
    if relation in evaluated_many_to_many_relations:
        return
    task = partial(
        _resolve_many_to_many_relation_in_loop,
        relation=relation,
        settings=settings,
        needed_setting_names=needed_settings_for_node,
        setting_value_dataset=dataset,
        dims=dims,
        sweeps=sweeps,
    )
    result = list(
        pool.map(task, range(n_tot)),
    )
    for (
        output_setting_name,
        function_argument_name_or_index,
    ) in relation.output_parameters.items():
        # Split here the data
        values = [value[function_argument_name_or_index] for value in result]
        value_array = np.reshape(
            create_numpy_array_with_fixed_dimensions(
                np.array(values, dtype=object), tuple(dims)
            ),
            dims,
        )
        dataset[output_setting_name] = (tuple(sweeps_for_node), value_array)
    evaluated_many_to_many_relations.add(relation)


def _resolve_many_to_many_relation_in_loop(
    ii, relation, settings, needed_setting_names, setting_value_dataset, dims, sweeps
):
    """
    Args:
        ii: Index of the current sweep for dimensions of the sweeps needed by the target relation.
        relation: Relation to evaluate.
        settings: A settings object which relations have been evaluated.
        needed_setting_names: A list of setting names that are needed for evaluating the given relation.
        setting_value_dataset: Dataset that contains all the values for already evalauted settings.
        dims: Dimensions for sweeps for the given relation.
        sweeps: All sweeps.
        mapped_setting_names: Setting names directly needed for evaluation.

    Returns:
        A new dataset which contains the previously evalauted Setting values
    """
    relation = copy.copy(relation)
    # settings = settings.copy()
    current_ind = np.unravel_index(ii, dims)
    current_sweeps = [
        setting_name for setting_name in needed_setting_names if setting_name in sweeps
    ]
    for setting_name in needed_setting_names:
        if setting_name in sweeps:
            # Get the setting value from the sweep
            sweep_ind = int(current_ind[current_sweeps.index(setting_name)])
            setattr(settings, setting_name, sweeps[setting_name][sweep_ind])
        if setting_name in setting_value_dataset:
            # Get the setting value from previously evaluated setting
            list_descriptor = tuple(
                [
                    current_ind[ii]
                    for ii, sweep_name in enumerate(current_sweeps)
                    if sweep_name in setting_value_dataset[setting_name].dims
                ]
            )
            setting_value = setting_value_dataset[setting_name].values[list_descriptor]
            setattr(settings, setting_name, setting_value)
    relation.resolve(settings)
    return relation.evaluated_value


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
    execution_settings,
):
    """
    Evaluates relations for a given setting and adds the result to the dataset.
    """
    task = partial(
        _resolve_in_loop,
        relation=setting.relation,
        settings=settings,
        needed_setting_names=needed_settings_for_node,
        setting_value_dataset=dataset,
        dims=dims,
        sweeps=sweeps,
        mapped_setting_names=mapped_setting_names,
    )
    value_array = np.reshape(
        create_numpy_array_with_fixed_dimensions(
            np.array(
                list(
                    pool.map(task, range(n_tot)),
                ),
                dtype=object,
            ),
            tuple(dims),
        ),
        dims,
    )
    dataset[setting.name] = (tuple(sweeps_for_node), value_array)


def _resolve_in_loop(
    ii,
    relation,
    settings,
    needed_setting_names,
    setting_value_dataset,
    dims,
    sweeps,
    mapped_setting_names,
):
    """
    Args:
        ii: Index of the current sweep for dimensions of the sweeps needed by the target relation.
        relation: Relation to evaluate.
        settings: A settings object which relations have been evaluated.
        needed_setting_names: A list of setting names that are needed for evaluating the given relation.
        setting_value_dataset: Dataset that contains all the values for already evalauted settings.
        dims: Dimensions for sweeps for the given relation.
        sweeps: All sweeps.
        mapped_setting_names: Mapped setting names for the given relation.
    Returns:
        A new dataset which contains the previously evalauted Setting values
    """
    relation = copy.copy(relation)
    # settings = settings.copy()
    current_ind = np.unravel_index(ii, dims)
    current_sweeps = [
        setting_name for setting_name in needed_setting_names if setting_name in sweeps
    ]
    for setting_name in mapped_setting_names:
        if setting_name in sweeps:
            # Get the setting value from the sweep
            sweep_ind = int(current_ind[current_sweeps.index(setting_name)])
            setattr(settings, setting_name, sweeps[setting_name][sweep_ind])
        if setting_name in setting_value_dataset:
            # Get the setting value from previously evaluated setting
            setting_name_dims = setting_value_dataset[setting_name].dims
            list_descriptor = tuple(
                [
                    current_ind[ii]
                    for ii, sweep_name in enumerate(current_sweeps)
                    if sweep_name in setting_name_dims
                ]
            )
            setting_value = setting_value_dataset[setting_name].values[list_descriptor]
            setattr(settings, setting_name, setting_value)

    relation.resolve(settings)
    return relation.evaluated_value


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


class DataSetWrapper():
    """
    Wrapper around dataset that can be used with multprocessing managers.
    """
    def __init__(self, dataset: xr.Dataset):
        self.dataset=dataset

    def __getitem__(self, key):
        return self.dataset[key]

    def __setitem__(self, key, value):
        self.dataset[key] = value
