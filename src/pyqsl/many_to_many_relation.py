"""
Defines many-to-many relations.

Classes:
  * ManyToManyRelation
"""

import dataclasses
from typing import Any

from .relation import Function, Relation


@dataclasses.dataclass
class ManyToManyRelation(Function):
    """
    Many-to-many relations represent mapping from N input arguments to
    M output arguments.
    """

    output_parameters: dict[str, str | int] = dataclasses.field(default_factory=dict)

    def get_mapped_output_setting_names(self) -> set[str]:
        """
        Returns the output setting names for the relation.
        """
        return set(self.output_parameters.keys())

    def __hash__(self) -> int:
        return self.identifier


@dataclasses.dataclass
class EvaluatedManyToManyRelation(Relation):
    """
    A place-holder for evaluated many-to-many relation.
    """

    source: str = ""

    def evaluate(self, **parameter_values) -> Any:
        """
        Does nothing when evaluated.
        """

    def __str__(self) -> str:
        return f"source: {self.source}"
