"""
Create a settings object for pyqsl.

Classes:

    * Setting
    * Settings
"""
from __future__ import annotations

import collections
import copy
import dataclasses
import logging
import types
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import networkx as nx

logger = logging.getLogger(__name__)
# pyright: reportPropertyTypeMismatch=false
# pylint: disable=cyclic-import


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
    _relation: Optional["Relation"] = dataclasses.field(
        init=False, repr=False, default=None
    )
    relation: Optional["Relation"] = None
    use_relation: bool | None = None  # use_relation has to come after relation.
    # The order of the values matter. _dimensions has to be first so that it is
    # initialized before the value is changed by the setter.
    _dimensions: list[str] = dataclasses.field(
        init=False, repr=False, default_factory=list
    )
    dimensions: list[str] = dataclasses.field(default_factory=list)
    _value: Optional[Any] = dataclasses.field(init=False, repr=False, default=None)
    description: Optional[str] = ""

    def __post_init__(self):
        if self.use_relation is None:
            if self.relation is None:
                self.use_relation = False
            else:
                self.use_relation = True

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
        return self.relation is not None and bool(self.use_relation)

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
            if self.use_relation is not None:
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
            return

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

    Note that settings cannot use the names of methods in :class:`Settings`.
    """

    _fields: dict[str, Setting] = dataclasses.field(default_factory=dict)
    _many_to_many_relations: list[ManyToManyRelation] = dataclasses.field(
        default_factory=list, repr=False, init=False
    )

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
            if setting.description:
                splits = setting.description.split(".", 1)
                n_letters = 40
                description = splits[0] + "." if len(splits) > 1 else splits[0]
                description = (
                    description[:n_letters] + ".."
                    if len(description) > (n_letters + 2)
                    else description
                )
                output = output + " - " + description
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

    def __getattr__(self, name):
        """
        If the target setting is not found, creates a new setting.

        At the moment, disabled. Works, but supports too error-prone programming.

        Args:
            name: Name of the Setting that is searched.
        """
        raise AttributeError(f'Trying to get setting "{name}" which was not found.')
        # if name.startswith('_'):
        #     raise AttributeError()
        # self.__setattr__(name, None)
        # return self[name]

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

    def __setitem__(self, key: str, value: Any | Setting):
        setattr(self, key, value)

    def add_many_to_many_relation(self, relation: ManyToManyRelation):
        """
        Adds a many-to-many relation to the settings.

        Args:
            relation: Many-to-many relation to be added.
        """
        self._many_to_many_relations.append(relation)

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
        many_to_many_relation_map = self.get_many_to_many_relation_map()
        relation_graph = self.get_relation_hierarchy()
        evaluated_many_to_many_relations = set()
        nodes_with_relation = []
        if not is_acyclic(relation_graph):
            raise ValueError("Cyclic relations detected.")
        # Resolve values
        nodes = list(nx.topological_sort(relation_graph))
        for node in nodes:
            setting = self[node]
            if setting.has_active_relation() and not isinstance(
                setting.relation, EvaluatedManyToManyRelation
            ):
                setting.relation.resolve(self)
                nodes_with_relation.append(node)
            elif setting.name in many_to_many_relation_map:
                relation = many_to_many_relation_map[setting.name]
                if relation not in evaluated_many_to_many_relations:
                    relation.resolve(self)
                    for (
                        output_setting_name,
                        function_argument_name_or_index,
                    ) in relation.output_parameters.items():
                        dummy_relation = EvaluatedManyToManyRelation(
                            parameters={}, source=str(relation)
                        )
                        if relation.evaluated_value is not None:
                            dummy_relation.evaluated_value = relation.evaluated_value[
                                function_argument_name_or_index
                            ]
                        else:
                            dummy_relation.evaluated_value = None
                        self[output_setting_name].relation = dummy_relation
                        nodes_with_relation.append(output_setting_name)
                    evaluated_many_to_many_relations.add(relation)

        dimensions = set()
        for setting in self:
            if setting.dimensions:
                dimensions.update(setting.dimensions)
        intersection = dimensions & set(nodes_with_relation)
        if intersection:
            raise ValueError(
                "Settings that are used as dimensions cannot have relations. "
                + f"These are the settings {list(intersection)}."
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
                    relation_graph.add_edge(
                        dependent_setting, setting.name, name=str(relation)
                    )
        for relation in self._many_to_many_relations:
            dependent_settings = relation.get_mapped_setting_names()
            for dependent_setting in dependent_settings:
                if dependent_setting not in relation_graph:
                    relation_graph.add_node(dependent_setting)
                for output_setting in relation.get_mapped_output_setting_names():
                    if output_setting not in relation_graph:
                        relation_graph.add_node(output_setting)
                    elif self[output_setting].has_active_relation() and not isinstance(
                        self[output_setting].relation, EvaluatedManyToManyRelation
                    ):
                        raise ValueError(
                            f'Output "{output_setting}" of a many-to-many relation "{relation}"'
                            + f' overlaps with an existing relation "{self[output_setting].relation}".'
                        )
                    relation_graph.add_edge(
                        dependent_setting, output_setting, name=(relation)
                    )
        return relation_graph

    def get_many_to_many_relation_map(self) -> dict[str, ManyToManyRelation]:
        """
        Returns a mapping from setting names to linked many-to-many relations.
        """
        many_to_many_relation_map = {}
        for relation in self._many_to_many_relations:
            output_parameters = relation.get_mapped_output_setting_names()
            for output_parameter in output_parameters:
                many_to_many_relation_map[output_parameter] = relation
        return many_to_many_relation_map

    def draw_relation_hierarchy(self):
        """
        Draws the current relation hierarchy.
        """
        relation_graph = self.get_relation_hierarchy()
        pos = _hierarchical_layout(relation_graph, orientation="v")
        # pos=nx.spring_layout(relation_graph)
        nx.draw(relation_graph, pos, with_labels=True)
        labels = {node: node for node in relation_graph.nodes}
        edge_labels = nx.get_edge_attributes(relation_graph, "name")
        nx.draw_networkx_labels(relation_graph, pos, labels, font_size=12)
        nx.draw_networkx_edge_labels(relation_graph, pos, edge_labels=edge_labels)

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

    def get_needed_settings(
        self, setting: Setting, relation_hierarchy: Optional[nx.DiGraph] = None
    ) -> list[str]:
        """
        Returns the names for settings needed to evaluate the given setting.

        Args:
            setting: Setting for which needed settings are searched for.
            relation_hierarchy: Relation hierarchy for the settings tree. If None, a new hierarchy is built.

        Returns:
            List of all settings needed to evaluate the given setting.
        """
        if relation_hierarchy is None:
            relation_hierarchy = self.get_relation_hierarchy()
        if setting.name not in relation_hierarchy:
            return []
        ancestors = nx.ancestors(relation_hierarchy, setting.name)
        return sorted(ancestors)

    def copy(self) -> Settings:
        """
        Creates a copy of self.

        Indidivual Setting objects are copied but their values are not.
        """
        new_settings = Settings()
        for setting in self:
            setattr(new_settings, setting.name, dataclasses.replace(setting))
            # There is a weird functionality in dataclasses with init=False, which prevents
            # dimensions from being copied.
            new_settings[setting.name].dimensions = copy.copy(setting.dimensions)
            new_settings[setting.name].use_relation = setting.use_relation
        for relation in self._many_to_many_relations:
            new_settings.add_many_to_many_relation(dataclasses.replace(relation))
        return new_settings


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


def _hierarchical_layout(
    graph: nx.DiGraph[str],
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    orientation: str = "h",
) -> dict[str, tuple[float, float]]:
    """
    Layout for plotting DiGraph by generations.

    Partial credits to chatGPT which provided a completely incorrect solution.

    Args:
        graph: Graph for which the node positions are calculated.
        dx: Space between the nodes in the same generation.
        dy: Space between the nodes in different genrations.
        orientation: If 'h', the same generation is horizontal, otherwise vertical.

    Returns:
        A dict which keys are the nodes and the values are the coordinates for the nodes.
    """
    pos = {}
    nbr_y = 0
    nbr_x = 0
    for generation in nx.topological_generations(graph):
        nbr_y = nbr_y + 1
        nbr_x = max(nbr_x, len(generation))
    if dx is None:
        dx = 1 / (nbr_x + 1)
    if dy is None:
        dy = 1 / (nbr_y + 1)

    # Need to call again because the result is a generator.
    y_coord = dy
    for generation in nx.topological_generations(graph):
        x_coord = dx
        y_coord = y_coord + dy
        for node in generation:
            if orientation.lower() == "h":
                pos[node] = (x_coord, y_coord)
            else:
                pos[node] = (y_coord, x_coord)
            x_coord = x_coord + dx
    return pos


from .many_to_many_relation import (  # pylint: disable=wrong-import-position
    EvaluatedManyToManyRelation,
    ManyToManyRelation,
)

# Avoid cyclic import
from .relation import Equation, Relation  # pylint: disable=wrong-import-position
