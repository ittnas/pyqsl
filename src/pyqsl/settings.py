"""
Create a settings object for pyqsl.

Classes:

    Settings
"""
import dataclasses
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Relation:
    pass


@dataclass
class Setting:
    name: str
    value: Any = None
    unit: str = ""
    relation: Optional[Relation] = None

    def __add__(self, o):
        return self.value + o

    __radd__ = __add__

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        return self.name == other.name


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
        # fields = self.__dict__
        fields = self._fields
        for field, value in fields.items():
            output = output + f"{field}: {value.value}"
            if value.unit:
                output = output + f" {value.unit}"
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
        # setting_dict = dataclasses.asdict(self)
        # name_value_dict: dict[str, Any] = {}
        # for name, setting in setting_dict.items():
        #     name_value_dict[name] = setting.value
        # #return name_value_dict
        # return name_value_dict
        return {name: setting.value for name, setting in self._fields.items()}
        # return self._fields
