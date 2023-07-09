"""
Create a settings object for pyqsl.

Classes:

    Settings
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional, Union

import networkx as nx


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
    """

    name: str
    value: Any
    unit: str = ""
    relation: Optional["Relation"] = None
    use_relation: bool = False
    _relation: Optional["Relation"] = dataclasses.field(
        init=False, repr=False, default=None
    )
    _value: Optional[Any] = dataclasses.field(init=False, repr=False, default=None)

    def __add__(self, other):
        return self.value + other

    __radd__ = __add__

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        return self.name == other.name

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


@dataclass
class Settings:
    """
    A class that contains settings for the simulation.
    """

    _fields: dict[str, Setting] = dataclasses.field(default_factory=dict)

    def __str__(self) -> str:
        return self._to_string()

    def __repr__(self) -> str:
        return self._to_string()

    def _to_string(self):
        output: str = ""

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
            ValueError if cyclic relations
        """
        # Build relation hierarchy
        relation_graph = self._build_relation_hierarchy()
        if not is_acyclic(relation_graph):
            raise ValueError("Cyclic relations detected.")
        # Resolve values
        nodes = list(nx.topological_sort(relation_graph))
        for node in nodes:
            setting = self[node]
            if setting.has_active_relation():
                setting.relation.resolve(self)
        return nodes
        #

    def _build_relation_hierarchy(self) -> nx.DiGraph:
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
