"""
Create a settings object for pyqsl.

Classes:

    Settings
"""
import dataclasses
from dataclasses import dataclass
from typing import Any, Optional, Union
import networkx as nx


@dataclass
class Setting:
    name: str
    value: Any
    unit: str = ""
    relation: Optional["relation.Relation"]
    use_relation: bool = False
    _relation: Optional["relation.Relation"] = dataclasses.field(init=False, repr=False, default=None)
    _value: Optional[Any] = dataclasses.field(init=False, repr=False, default=None)

    def __add__(self, o):
        return self.value + o

    __radd__ = __add__

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        return self.name == other.name

    def has_active_relation(self)-> bool:
        """
        Returns True if relation is not None and use_relation is True.
        """
        return self.relation is not None and self.use_relation

    @property
    def relation(self) -> "relation.Relation":
        return self._relation

    @relation.setter
    def relation(self, value: Union[str, "relation.Relation"]):
        if type(value) is property:
            # initial value not specified, use default
            value = None
        if isinstance(value, str):
            value = Equation(equation=value)
        self._relation = value
        if value is not None:
            self.use_relation = True

    @property
    def value(self) -> Any:
        return self.relation.evaluated_value if (self.has_active_relation() and self.relation.evaluated_value is not None) else self._value

    @value.setter
    def value(self, value: Any):
        if type(value) is property:
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
        
        #for field, value in fields.items():
        for setting in self:
            output = output + f"{setting.name}: {setting.value}"
            if setting.unit:
                output = output + f" {setting.unit}"
            if setting.relation is not None:
                output = output + f", {setting.relation} ({'on' if setting.use_relation else 'off'})"
            if setting.has_active_relation():
                output = output + f", value={setting._value}"
            output = output + "\n"
        return output

    def __setattr__(self, name, value):
        if isinstance(value, Setting):
            if value.name != name:
                raise ValueError(
                    f"Setting name ({value.name}) has to be equal to Settings field name ({name})."
                )
            super().__setattr__(name, value)
            if value.name not in self._fields:
                self._fields[value.name]: value
        else:
            if not name.startswith("_"):
                setting = Setting(name, value)
                self._fields[setting.name] = setting
                super().__setattr__(name, setting)
            else:
                super().__setattr__(name, value)

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
    def resolve_relations(self):
        """
        Resolves all the relations in the settings hierarchy.

        Raises:
            ValueError if cyclic relations
        """
        # Build relation hierarchy
        relation_graph = self._build_relation_hierarchy()
        if not is_acyclic(relation_graph):
            raise ValueError('Cyclic relations detected.')
        # Resolve values
        for node in nx.topological_sort(relation_graph):
            setting = self[node]
            if setting.has_active_relation():
                setting.relation.resolve(self)
        #

    def _build_relation_hierarchy(self) -> nx.DiGraph:
        settings_with_active_relation = [setting for setting in self if setting.has_active_relation()]
        relation_graph = nx.DiGraph()
        relation_graph.add_nodes_from([setting.name for setting in settings_with_active_relation])
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
    for cycle in nx.simple_cycles(graph):
        return False
    return True


from .relation import Equation  # Cyclic import
