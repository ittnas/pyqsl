"""
Classes for relations between settings.

Relations can be used to define a mathematical relation between two or more settings.
The relations for parameters that are not swept are resolved before the simulation loop.
The settings with relations that are part of the sweeps dependency tree will be resolved within
the sweep.

Classes:
    Relation
    Equation
    LookupTable
"""
import dataclasses
from typing import Union, Any, Optional
from .settings import Setting, Settings
from abc import ABC, abstractmethod
import numexpr as ne


@dataclasses.dataclass
class Relation(ABC):
    parameters: dict[str, Union[str, "Relation"]]
    _parameters: dict[str, Union[str, "Relation"]] = dataclasses.field(init=False, repr=False)
    evaluated_value: Optional[Any] = None

    @property
    def parameters(self) -> dict[str, Union[str, "Relation"]]:
        return self._parameters

    @parameters.setter
    def parameters(self, v: dict[str, Union[str, Setting, "Relation"]]) -> None:
        if type(v) is property:
            # initial value not specified, use default
            v = {}
        self._parameters = {}
        for key, value in v.items():
            if isinstance(value, Setting):
                self._parameters[key] = value.name
            else:
                self._parameters[key] = value

    @abstractmethod
    def evaluate(self, **parameter_values) -> Any:
        pass

    def get_mapped_setting_names(self) -> set[str]:
        """
        Iteratively fetches mapped settings for nested Relations.
        """
        mapped_setting_names = set()
        for setting_or_relation in self.parameters.values():
            if isinstance(setting_or_relation, Relation):
                mapped_setting_names.update(setting_or_relation.get_mapped_setting_names())
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


@dataclasses.dataclass
class Equation(Relation):
    equation: str = '0'

    def __post_init__(self):
        expr = ne.NumExpr(self.equation)
        for name in expr.input_names:
            if name not in self.parameters:
                self._parameters[name] = name

    def evaluate(self, **parameter_values):
        """
        Evaluates the equation using numexpr evaluation logic.

        Known issue: numexpr does not support string arguments.
        """
        expr = ne.evaluate(self.equation, local_dict=parameter_values, global_dict={})
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
    pass
