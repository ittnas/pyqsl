import dataclasses
import json

import pytest

import pyqsl


def test_create_setting():
    empty_setting = pyqsl.Setting("frequency", unit="Hz")
    assert empty_setting.value is None
    setting = pyqsl.Setting("amplitude", 4)
    setting = pyqsl.Setting("amplitude", 4, unit="V")
    assert setting.value == 4
    assert setting.unit == "V"
    assert setting.name == "amplitude"


def test_create_setting_with_no_name():
    empty_setting = pyqsl.Setting(unit="Hz")
    assert empty_setting.value is None
    assert empty_setting.name is None
    empty_setting.name = "frequency"
    assert empty_setting.name == "frequency"


def test_create_settings():
    settings = pyqsl.Settings()
    settings.amplitude = 3
    assert settings.amplitude.value == 3
    assert settings.amplitude.name == "amplitude"
    setting = pyqsl.Setting("amplitude")
    settings.amplitude = setting
    assert settings.amplitude.value is None
    settings.amplitude = 5
    assert settings.amplitude.value == 5

    setting_with_no_name = pyqsl.Setting()
    settings.frequency = setting_with_no_name
    assert settings.frequency.name == "frequency"


def test_overloading_add():
    settings = pyqsl.Settings()
    settings.amplitude = 5
    assert 3 + settings.amplitude == 8
    assert settings.amplitude + 3 == 8
    assert settings.amplitude + settings.amplitude == 10
    assert settings.amplitude + settings.amplitude != 9


def test_overloading_other_operators():
    settings = pyqsl.Settings()
    settings.a = 5
    settings.b = 2
    settings.c = 0.5

    assert (settings.a - settings.b) == (settings.a.value - settings.b.value)
    settings.c = settings.a * settings.b
    assert settings.c == 10
    settings.c = settings.a**settings.b
    assert settings.c == 25
    settings.c = settings.a / settings.b
    assert settings.c == 2.5
    settings.c = settings.a // settings.b
    assert settings.c == 2
    settings.c = settings.a % settings.b
    assert settings.c == 1
    settings.c = settings.a << settings.b
    assert settings.c == 20
    settings.c = settings.a >> settings.b
    assert settings.c == 1
    settings.c = settings.a & settings.b
    assert settings.c == 0
    settings.c = settings.a | settings.b
    assert settings.c == 7
    settings.c = settings.a ^ settings.b
    assert settings.c == 7
    settings.c = settings.a > settings.b
    assert settings.c
    assert (settings.a < settings.b) == (settings.a.value < settings.b.value)
    settings.c = settings.a >= settings.b
    assert settings.c
    settings.c = settings.a <= settings.b


def test_adding_setting_with_wrong_name():
    settings = pyqsl.Settings()
    settings.amplitude = 5
    setting = pyqsl.Setting("frequency", 4)
    with pytest.raises(ValueError):
        settings.amplitude = setting


def test_converting_to_dict(settings):
    settings_dict = settings.to_dict()
    assert settings_dict["amplitude"] == 4


def test_adding_to_dict(settings):
    setting_dict = {settings.amplitude: [0, 1, 2]}
    assert setting_dict["amplitude"] == [0, 1, 2]
    assert setting_dict[settings.amplitude] == [0, 1, 2]
    setting_dict["amplitude"] = [0, 1]
    assert setting_dict[settings.amplitude] == [0, 1]


def test_relation_property(settings):
    assert settings.amplitude.use_relation is False
    settings.amplitude.relation = pyqsl.Equation(equation="a+1")
    assert settings.amplitude.use_relation is True
    # And reverse
    settings.amplitude = 5
    assert settings.amplitude.use_relation is False


def test_getitem(settings):
    assert settings["amplitude"] == settings.amplitude


def test_saving_as_json(settings):
    settings_dict = dataclasses.asdict(settings)
    json_dump = json.dumps(settings_dict)


def test_setting_value(settings):
    assert settings.amplitude.unit == "V"
    settings.amplitude = 2
    assert settings.amplitude.unit == "V"
    settings.phase = 1.41
    assert settings.phase.value == 1.41


def test_copy():
    settings = pyqsl.Settings()
    settings.a = [0]
    settings.b = 1
    settings.c = None
    copied = settings.copy()
    settings.c.relation = pyqsl.Equation(equation="b + 1")
    assert copied.b.value == settings.b.value
    copied.b = 2
    assert copied.b.value != settings.b.value
    assert copied.a.value[0] == settings.a.value[0]
    settings.a.value[0] = 1
    assert copied.a.value[0] == settings.a.value[0]
    assert copied.c.relation is None
    assert settings.c.relation is not None


def test_indexing():
    settings = pyqsl.Settings()
    settings.a = [0, 1, 2]
    settings.b = 2
    assert settings.a[0] == 0
    assert settings.a.value[0] == settings.a[0]
    assert settings.a[2] + settings.a[1] == 3
    assert settings.b + settings.a[1] == 3
    assert not settings.b.use_relation


def test_dimensions():
    settings = pyqsl.Settings()
    settings.a = 2
    settings.b = [0, 1, 2]
    settings.c = [0, 1, 2, 3]
    assert settings.a.dimensions == []
    settings.a.dimensions = ["b", settings.b]
    assert settings.a.dimensions == ["b"]
    settings.a.dimensions = ["b", settings.b, settings.c]
    assert settings.a.dimensions == ["b", "c"]

    settings.a.dimensions = settings.b
    assert settings.a.dimensions == ["b"]
    with pytest.raises(TypeError):
        settings.a.dimensions = "abab"
    with pytest.raises(TypeError):
        settings.a.dimensions = [["abab"]]


def test_indexing():
    settings = pyqsl.Settings()
    settings["c"] = 2
    assert settings.c == 2
    settings["d"] = pyqsl.Setting(value=3, dimensions=["b"])
    assert settings.d.value == 3
    assert settings.d.dimensions == ["b"]


def test_init_with_relations():
    settings = pyqsl.Settings()
    settings.a = 2
    settings.b = pyqsl.Setting(relation="a")
    assert settings.b.use_relation == True
    settings.b = pyqsl.Setting(relation="a", use_relation=False)
    assert settings.b.use_relation == False
    settings.b.relation = "a"
    assert settings.b.use_relation == True


def test_many_to_many_relations():
    settings = pyqsl.Settings()

    def function(var1, var2):
        return var1, var2

    settings.a = 2
    settings.b = 3
    settings.c = None
    settings.d = None
    settings.e = pyqsl.Setting(relation="c + d")
    settings.add_many_to_many_relation(
        pyqsl.ManyToManyRelation(
            function=function,
            parameters={"var1": "a", "var2": "b"},
            output_parameters={"c": 0, "d": 1},
        )
    )
    settings.resolve_relations()
    assert settings.c == 2
    assert settings.d == 3
    assert settings.e == 5
    settings.resolve_relations()
    assert settings.c == 2
    assert settings.d == 3
    assert settings.e == 5
    pyqsl.run(None, settings)
    pyqsl.run(None, settings, sweeps={"a": [0, 1, 2]})
    pyqsl.run(None, settings, sweeps={"a": [0, 1, 2], "b": [3, 4]})


def test_many_to_many_relations_with_dict():
    settings = pyqsl.Settings()

    def function(var1, var2):
        return {"v1": var1, "v2": var2}

    settings.a = 2
    settings.b = 3
    settings.c = None
    settings.d = None
    settings.e = pyqsl.Setting(relation="c + d")
    settings.add_many_to_many_relation(
        pyqsl.ManyToManyRelation(
            function=function,
            parameters={"var1": "a", "var2": "b"},
            output_parameters={"c": "v1", "d": "v2"},
        )
    )
    settings.resolve_relations()
    assert settings.c == 2
    assert settings.d == 3
    assert settings.e == 5


def test_description():
    settings = pyqsl.Settings()
    settings.a = 2
    settings.a.description = "The value of a."
    assert settings.a.description == "The value of a."
    assert settings._to_string().endswith("The value of a.\n")
    settings.b = 2
    settings.b.description = "The value of b. This continues."
    assert settings._to_string().endswith("The value of b.\n")
