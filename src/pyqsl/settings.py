"""
Create a settings object for pyqsl.

Classes:

    Setting
    Settings
"""
from __future__ import annotations

import collections
import dataclasses
import logging
import types
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class Setting:
    """
    Describes a setting for the simulation.

    A settings are used as an input arguments for simulation task functions. They
    are basically name-value pairs, where name should correspond to the name of the
    input argument. Additinoanally, settings can contain meta-information, such as
    unit for the quantity.

    Some settings are not free parameters, but related to each other. This can be
    described by asigning a relation to the setting. If relation is is activated
    through setting use_relation=True, the value of the setting is the evaluated
    value of the relation instead (``self.relation.evaluated_value``).

    If the setting is vector-valued, giving another setting as a dimension through
    ``dimensions`` field indicates that the setting should be interpreted as the
    coordinate for the target. Note that setting that is used as a dimension cannot
    be swept directly or indirectly.
    """

    # pylint: disable=too-many-instance-attributes
    name: Optional[str] = None
    value: Optional[Any] = None
    unit: str = ""
    relation: Optional["Relation"] = None
    use_relation: bool = False
    dimensions: list[str] = dataclasses.field(default_factory=list)
    _relation: Optional["Relation"] = dataclasses.field(
        init=False, repr=False, default=None
    )
    _value: Optional[Any] = dataclasses.field(init=False, repr=False, default=None)
    _dimensions: list[str] = dataclasses.field(
        init=False, repr=False, default_factory=list
    )

    def __add__(self, other):
        return self.value + other

    __radd__ = __add__

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        return self.value * other

    __rmul__ = __mul__

    def __pow__(self, other):
        return self.value**other

    def __rpow__(self, other):
        return other**self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __floordiv__(self, other):
        return self.value // other

    def __rfloordiv__(self, other):
        return other // self.value

    def __mod__(self, other):
        return self.value % other

    def __rmod__(self, other):
        return other % self.value

    def __lshift__(self, other):
        return self.value << other

    def __rlshift__(self, other):
        return other << self.value

    def __rshift__(self, other):
        return self.value >> other

    def __rrshift__(self, other):
        return other >> self.value

    def __and__(self, other):
        return self.value & other

    __rand__ = __and__

    def __or__(self, other):
        return self.value | other

    __ror__ = __or__

    def __xor__(self, other):
        return self.value ^ other

    __rxor__ = __xor__

    def __lt__(self, other):
        return self.value < other

    __rlt__ = __lt__

    def __gt__(self, other):
        return self.value > other

    __rgt__ = __gt__

    def __ne__(self, other):
        if isinstance(other, str):
            return self.name != other
        if isinstance(other, Setting):
            return self.name != other.name
        return self.value > other

    def __le__(self, other):
        return self.value <= other

    __rle__ = __le__

    def __ge__(self, other):
        return self.value >= other

    __rge__ = __ge__

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        if isinstance(other, Setting):
            return self.name == other.name
        return self.value == other

    def __getitem__(self, key):
        return self.value[key]

    def has_active_relation(self) -> bool:
        """
        Returns True if relation is not None and use_relation is True.
        """
        return self.relation is not None and self.use_relation

    @property  # type: ignore[no-redef]
    def relation(self) -> "Relation" | None:
        """
        Returns the relation for the setting if set or None otherwise.
        """
        return self._relation

    @relation.setter
    def relation(self, value: Union[str, "Relation" | None]):
        if isinstance(value, property):
            # initial value not specified, use default
            value = None
        if isinstance(value, str):
            value = Equation(equation=value)  # type: ignore[call-arg]
        self._relation = value
        if value is not None:
            self.use_relation = True

    @property  # type: ignore[no-redef]
    def value(self) -> Any:
        """
        Returns the value of the setting.

        If there is an activate relation, i.e. ``self.has_active_relation()==True`` and
        the relation is evaluated, returns the evaluated value of the relation instead.
        """
        return (
            self.relation.evaluated_value  # type: ignore[union-attr]
            if (
                self.has_active_relation() and self.relation.evaluated_value is not None  # type: ignore[union-attr]
            )
            else self._value
        )

    @value.setter
    def value(self, value: Any):
        if isinstance(value, property):
            value = None
        self._value = value

    @property  # type: ignore[no-redef]
    def dimensions(self) -> list[str]:
        """
        Returns the dimensions for the setting.
        """
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value: Setting | Sequence[str | Setting]):
        """
        Sets the dimensions attribute.

        Args:
            value:
                A sequence of settings, settings names or a single setting used to initialize
                the dimensions list.
        Raises:
            TypeError if wrong datatype is used to initialize the setting.
        """
        if isinstance(value, property):
            value = []
        new_dimension_list: list[str] = []
        sequence_of_dimensions: Sequence[str | Setting]
        if isinstance(value, str):
            raise TypeError("Provide a list of strings instead of a string.")
        if isinstance(value, Setting):
            sequence_of_dimensions = [value]
        else:
            sequence_of_dimensions = value  # type: ignore[assignment]
        for dimension in sequence_of_dimensions:
            if isinstance(dimension, Setting):
                new_dimension_list.append(dimension.name)  # type: ignore[arg-type]
            elif isinstance(dimension, str):
                new_dimension_list.append(dimension)
            else:
                raise TypeError(
                    "Either setting or setting name must be used for dimensions."
                )

        # Remove duplicates
        seen: dict[str, None] = collections.OrderedDict()
        for element in new_dimension_list:
            seen[element] = None
        self._dimensions = list(seen.keys())


@dataclass
class Settings:
    """
    A collection of individual instances Setting objects.

    Settings are used as the input argument for pyqsl.run function and
    are the primary way of providing inputs for the simulation task.

    An empty ``Settings`` object is created as  ``settings = pyqsl.Settings()``.
    In order to add a setting for a parameter, one can do either of the following:
    *  ``settings.a = 2``
    *  ``a = Setting('a', 2); settings.a = a``

    The setting values can be accessed as ``settings['a']`` or ``settings.a``.
    The Setting objects can contain relations, which need to be resolved before
    their values can be accessed. This can be done as ``settings.resolve_relations()``.
    """

    _fields: dict[str, Setting] = dataclasses.field(default_factory=dict)

    def __str__(self) -> str:
        return self._to_string()

    def __repr__(self) -> str:
        return self._to_string()

    def _to_string(self):
        output = ""

        for setting in self:
            output = output + f"{setting.name}: {setting.value}"
            if setting.unit:
                output = output + f" {setting.unit}"
            if setting.relation is not None:
                output = (
                    output
                    + f", {setting.relation} ({'on' if setting.use_relation else 'off'})"
                )
            if setting.has_active_relation():
                output = (
                    output
                    + f", value={setting._value}"  # pylint: disable=protected-access
                )
            output = output + "\n"
        return output

    def __setattr__(self, name, value):
        """
        Sets attribute values.

        *  If name is a name of an existing setting, sets the value for that setting instead.
        *  If name does not exist and value is not an instance of Setting, creates a new setting with the given value.
        *  If name does not exist but value is a setting, adds that setting.

        Raises:
            ValueError if trying to add a new setting which name is different from the attribute name.
        """
        if isinstance(value, Setting):
            if value.name is None:
                value.name = name
            if value.name != name:
                raise ValueError(
                    f"Setting name ({value.name}) has to be equal to Settings field name ({name})."
                )
            super().__setattr__(name, value)
            self._fields[value.name] = value
        else:
            if name.startswith("_"):
                super().__setattr__(name, value)
            elif name in self:
                self[name].value = value
                self[name].use_relation = False
            else:
                setting = Setting(name, value)
                self._fields[setting.name] = setting
                super().__setattr__(name, setting)

    def to_dict(self) -> dict[str, Any]:
        """Name-value pair representation of the object

        Returns:
            Dictionary containing the setting names as keys and setting values as values.
        """
        return {setting.name: setting.value for setting in self}

    def to_namespace(self) -> types.SimpleNamespace:
        """
        Converts settings to SimpleNamespace with setting values as attributes.
        """
        return types.SimpleNamespace(**self.to_dict())

    def __iter__(self):
        yield from self._fields.values()

    def __getitem__(self, key: str):
        return self._fields[key]

    # def resolve_relations(self) -> Self: #  Needs python 3.11
    def resolve_relations(self) -> list[str]:
        """
        Resolves all the relations in the settings hierarchy.

        Returns:
            List of setting names with relations to resolve.

        Raises:
            ValueError if cyclic relations or settings which are used for dimensions
            have relations.
        """
        # Build relation hierarchy
        relation_graph = self.get_relation_hierarchy()
        nodes_with_relation = []
        if not is_acyclic(relation_graph):
            raise ValueError("Cyclic relations detected.")
        # Resolve values
        nodes = list(nx.topological_sort(relation_graph))
        for node in nodes:
            setting = self[node]
            if setting.has_active_relation():
                setting.relation.resolve(self)
                nodes_with_relation.append(node)
        dimensions = set()
        for setting in self:
            if setting.dimensions:
                dimensions.update(setting.dimensions)
        if dimensions & set(nodes_with_relation):
            raise ValueError(
                "Settings that are used as dimensions cannot have relations."
            )
        return nodes_with_relation

    def get_relation_hierarchy(self) -> nx.DiGraph:
        """
        Returns a directed graph describing the dependencies between relations.
        """
        settings_with_active_relation = [
            setting for setting in self if setting.has_active_relation()
        ]
        relation_graph = nx.DiGraph()
        relation_graph.add_nodes_from(
            [setting.name for setting in settings_with_active_relation]
        )
        for setting in settings_with_active_relation:
            relation = setting.relation
            dependent_settings = relation.get_mapped_setting_names()
            for dependent_setting in dependent_settings:
                if dependent_setting not in relation_graph:
                    relation_graph.add_node(dependent_setting)
                if dependent_setting != setting.name:
                    relation_graph.add_edge(dependent_setting, setting.name)
        return relation_graph

    def get_dependent_setting_names(
        self, setting: Setting, relation_hierarchy: Optional[nx.DiGraph] = None
    ) -> list[str]:
        """
        Returns the names for settings dependent on the given setting.

        Args:
            setting: Setting for which dependent settings are searched for.
            relation_hierarchy: Relation hierarchy for the settings tree. If None, a new hierarchy is built.

        Returns:
            List of all settings depending on the given setting.
        """
        if relation_hierarchy is None:
            relation_hierarchy = self.get_relation_hierarchy()
        if setting.name not in relation_hierarchy:
            return []
        descendants = nx.descendants(relation_hierarchy, setting.name)
        return sorted(descendants)

    def copy(self):
        """
        Creates a copy of self.

        Indidivual Setting objects are copied but their values are not.
        """
        settings = Settings()
        for setting in self:
            setattr(settings, setting.name, dataclasses.replace(setting))
        return settings


def is_acyclic(graph: nx.DiGraph) -> bool:
    """
    Checks if graph is cyclic.

    Args:
        graph: graph to be checked for cycles

    Returns:
        True if acyclic, False if cyclic.
    """
    for _ in nx.simple_cycles(graph):
        return False
    return True


# Avoid cyclic import
from .relation import Equation, Relation  # pylint: disable=wrong-import-position
