"""
Classes for relations between settings.

Relations can be used to define a mathematical relation between two or more settings.
The relations for parameters that are not swept are resolved before the simulation loop.
The settings with relations that are part of the sweeps dependency tree will be resolved within
the sweep.

Classes:
    * Relation
    * Equation
    * LookupTable
    * Function
"""

import copy
import dataclasses
import inspect
import itertools
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

import numexpr as ne
import numpy as np
from scipy.interpolate import interpn

from .settings import Setting

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Relation(ABC):
    """
    Relation represents a mathematical relation between different setting values in the settings hierarchy.

    The relations are represented through the attribute parameters, which maps the parameters used in the
    relation to other setting names, or alternatively, to other relations. How the related parameters are used
    is controlled by the evaluate-method, which is called during relation resolution. The evaluate method can be
    overridden and used to customize the behaviour of the relation.

    Note that circular relations are not supported, i.e. a setting ``a`` cannot depend on setting ``b`` that
    in turn depends on setting ``a`` as that would lead to infinite loop. However, the relation for setting ``a``
    can depend on ``a``. In this case the original value of ``a`` is used for relation resolution.

    Attributes:
        parameters: A mapping from parameter names used in the relation to setting names.
        evaluated_value:
            The resulting value of the relation evaluation. None implies that the relation has not been
            resolved.
    """

    parameters: dict[str, Union[str, "Relation"]]
    _parameters: dict[str, Union[str, "Relation"]] = dataclasses.field(
        init=False, repr=False
    )
    evaluated_value: Optional[Any] = None
    identifier: int = dataclasses.field(
        default_factory=itertools.count().__next__, init=False, repr=False
    )

    @property  # type: ignore[no-redef]
    def parameters(self) -> dict[str, Union[str, "Relation"]]:
        """
        Returns the parameter mapping for the relation.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, value: dict[str, Union[str, Setting, "Relation"]]) -> None:
        if isinstance(value, property):
            # initial value not specified, use default
            value = {}
        self._parameters = {}
        for key, dict_value in value.items():
            if isinstance(dict_value, Setting):
                self._parameters[key] = (
                    dict_value.name if dict_value.name is not None else "none"
                )
            else:
                self._parameters[key] = dict_value

    @abstractmethod
    def evaluate(self, **parameter_values) -> Any:
        """
        This abstract method should be overridden by deriving
        classes. The main evaluation logic of the relation should be
        implemented here.
        """

    def get_mapped_setting_names(self) -> set[str]:
        """
        Iteratively fetches mapped settings for nested Relations.
        """
        mapped_setting_names = set()
        for setting_or_relation in self.parameters.values():
            if isinstance(setting_or_relation, Relation):
                mapped_setting_names.update(
                    setting_or_relation.get_mapped_setting_names()
                )
            else:
                mapped_setting_names.add(setting_or_relation)
        return mapped_setting_names

    def resolve(self, settings):
        """
        This function contains the relation resolution logic.

        Fetches the parameter values from settings, assuming that the relations for
        all the dependent parameters have been resolved.

        Args:
            settings: settings hierarchy that is used for looking up parameter values.
        """
        parameter_values = {}
        for parameter, setting_name_or_relation in self.parameters.items():
            if isinstance(setting_name_or_relation, Relation):  # Nested relation
                setting_name_or_relation.resolve(settings)
                parameter_values[parameter] = setting_name_or_relation.evaluated_value
            else:
                parameter_values[parameter] = settings[setting_name_or_relation].value
        value = self.evaluate(**parameter_values)
        self.evaluated_value = value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, type(self)):
            return self.identifier == other.identifier
        return False


class RelationEvaluationError(Exception):
    """
    Error related to relation evaluation.
    """


@dataclasses.dataclass
class Equation(Relation):
    """
    Implements a simple mathematical equation between
    setting values using numexpr library.

    Examples:

    * ``eq = Equation(equation='a + b')`` creates an equation that sums up values
    of parameters ``a`` and ``b``. An implicit mapping of parameters ``a`` and ``b`` to
    a settings called ``a`` and ``b`` is created.

    * ``eq = Equation(equation='a + b', parameters={'a': 'amplitude', 'b': 'frequency'})`` maps
    parameters ``a`` and ``b`` to settings ``amplitude`` and ``frequency``.

    Attributes:
        equation:
            The implemented equation in string format. The equation represented by the string
            has to be interpretable by numexpr. By default the equation just returns 0.
    """

    equation: str = "0"

    def __post_init__(self):
        """
        Adds missing parameters from the equation to self.parameters.
        Assumes direct mapping between parameter names and settings.
        """
        expr = ne.NumExpr(self.equation)
        for name in expr.input_names:
            if name not in self.parameters:
                self._parameters[name] = name
                logger.debug('Adding missing parameter "%s".', name)

    def evaluate(self, **parameter_values):
        """
        Evaluates the equation using numexpr evaluation logic.

        Known issue: numexpr does not support string arguments.
        """
        try:
            expr = ne.evaluate(
                self.equation, local_dict=parameter_values, global_dict={}
            )
        except ValueError as err:
            values = {
                self.parameters[parameter]: value
                for parameter, value in parameter_values.items()
            }
            raise ValueError(
                (
                    f"An error occurred when evaluating equation '{str(self)}'"
                    f"for mapped settings {list(self.parameters.values())} with"
                    f"values { values }."
                )
            ) from err
        try:
            # ne creates 0-d arrays from scalars. Try to convert back.
            output = expr.item()
        except ValueError:
            output = expr
        return output

    def __str__(self):
        return f"Eq: {self.equation}"


@dataclasses.dataclass
class LookupTable(Relation):
    """
    Implements a lookup table for parameter values.

    Attributes:
        data:
        coordinates:

    Examples:
    ``LookupTable(data=[4, 0, 4], coordinates={'x': [-2, 0, 2]})``

    ``LookupTable(data=[4, 0, 4], coordinates={'x': [-2, 0, 2]}, parameters={'x': 'amplitude')``

    ``
        LookupTable(
        data=np.ones(3,2), coordinates={'x': [-2, 0, 2], 'y': [0, 1]}, parameters={'x': 'amplitude', 'y': 'frequency'
        )
    ``
    """

    data: Any = None
    coordinates: Optional[dict[str, Any]] = None

    def __post_init__(self):
        """
        Adds missing parameters from the lookup coordinates to self.parameters.

        Assumes direct mapping between parameter names and settings. Also checks that dimensions are consistent.

        Raises:
            ValueError if data and coordinate dimensions do not match.
        """
        for name in self.coordinates:
            if name not in self.parameters:
                self._parameters[name] = name
                logger.debug('Adding missing parameter "%s".', name)

        data_array = np.array(self.data, dtype=object)
        data_shape = data_array.shape
        coordinate_shape = tuple(
            len(coordinate_values) for coordinate_values in self.coordinates.values()
        )
        if data_shape != coordinate_shape:
            raise ValueError(
                f"Data shape is different from coordinate dimensions, {data_shape} != {coordinate_shape}"
            )

    def evaluate(self, **parameter_values):
        """
        Evaluates the lookup table.
        """
        point = [parameter_values[coordinate] for coordinate in self.coordinates]
        points = list(self.coordinates.values())
        values = self.data
        result = interpn(points, values, point)

        try:
            # interpn creates 0-d arrays from scalars. Try to convert back.
            output = result.item()
        except ValueError:
            output = result

        return output

    def __str__(self):
        output = "LookupTable for " + ", ".join(self.coordinates)
        return output


@dataclasses.dataclass
class Function(Relation):
    """
    Uses an arbitrary function for a relation.

    Attributes:
        function: Task function.
        function_arguments: Default keyword arguments for the function.

    Examples:
    ``Function(function=my_custom_function)``
    """

    function: Optional[Callable] = None
    function_arguments: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """
        Adds the argument names from the function signature as the parameters for the relation.
        """
        signature = inspect.signature(self.function)
        parameter_names = [
            param.name
            for param in signature.parameters.values()
            if param.default == inspect.Parameter.empty
        ]
        for name in parameter_names:
            if (name not in self.parameters) and (name not in self.function_arguments):
                self._parameters[name] = name
                logger.debug('Adding missing parameter "%s".', name)

    def evaluate(self, **parameter_values):
        """
        Evaluates the function.
        """
        arguments = copy.copy(self.function_arguments)
        arguments.update(parameter_values)
        return self.function(**arguments)

    def __str__(self):
        return f"F: {self.function.__name__}"
