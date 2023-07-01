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
    parameters: dict[str, str]
    _parameters: dict[str, str] = dataclasses.field(init=False, repr=False)
    _evaluated_value: Optional[Any] = None

    @property
    def parameters(self) -> dict[str, str]:
        return self._parameters

    @parameters.setter
    def parameters(self, v: dict[str, Union[str, Setting]]) -> None:
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
    def evaluate(self) -> Any:
        pass

    def resolve(self, settings):
        """
        This function contains the relation resolution logic.

        Fetches the values for variables by iteratively invoking
        resolve on dependant variables.

        Args:
            settings: settings hierarchy that is used for looking up parameter values.
        """
        pass


@dataclasses.dataclass
class Equation(Relation):
    equation: str = 'x'

    def __post_init__(self):
        expr = ne.NumExpr(self.equation)
        for name in expr.input_names:
            if name == 'x':  # Referst to self
                continue
            elif name not in self.parameters:
                self._parameters[name] = name

    def evaluate(self, parameter_values: dict[str, Any]):
        expr = ne.evaluate(self.equation, local_dict=parameter_values, global_dict={})
        return expr

    def _detect_parameters():
        pass


@dataclasses.dataclass
class LookupTable(Relation):
    pass
