import numpy as np
import pytest

import pyqsl
from pyqsl.relation import RelationEvaluationError


def test_that_abstract_relation_cannot_be_created():
    with pytest.raises(TypeError):
        relation = pyqsl.relation.Relation()


def test_create_equation():
    equation = pyqsl.relation.Equation(equation="a+1", parameters={"a": "b"})
    assert equation.parameters == {"a": "b"}
    assert equation.equation == "a+1"
    equation = pyqsl.relation.Equation()
    assert equation.parameters == {}
    assert equation.equation == "0"
    setting = pyqsl.Setting("b")
    equation = pyqsl.relation.Equation(equation="a+1", parameters={"a": setting})
    assert equation.parameters == {"a": "b"}


def test_paremeter_not_found_error():
    equation = pyqsl.relation.Equation(equation="a+1")
    assert equation.parameters == {"a": "a"}


def test_equation_evaluation():
    equation = pyqsl.relation.Equation(equation="a+1", parameters={"a": "b"})
    assert equation.evaluate(a=20) == 21
    equation = pyqsl.relation.Equation(equation="a*2 + c")
    assert equation.evaluate(a=2, c=1) == 5


def test_build_relation_hierarchy(settings_with_relations):
    graph = settings_with_relations.get_relation_hierarchy()
    assert set(graph.nodes) == set(["a", "b", "c", "e"])
    assert set(graph.edges) == {("a", "e"), ("b", "a"), ("c", "a"), ("c", "b")}


def test_that_self_reference_is_not_added(settings_with_relations):
    settings_with_relations.d.relation = "d"
    graph = settings_with_relations.get_relation_hierarchy()
    assert set(graph.nodes) == set(["a", "b", "c", "e", "d"])
    assert set(graph.edges) == {("a", "e"), ("b", "a"), ("c", "a"), ("c", "b")}


def test_is_acyclic():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.b = 2
    settings.a.relation = "b"
    settings.b.relation = "a"
    graph = settings.get_relation_hierarchy()
    assert pyqsl.settings.is_acyclic(graph) is False
    settings.b.relation = None
    graph = settings.get_relation_hierarchy()
    assert pyqsl.settings.is_acyclic(graph) is True


def test_resolve_relations_with_cyclic_graph():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.b = 2
    settings.a.relation = "b"
    settings.b.relation = "a"
    with pytest.raises(ValueError):
        settings.resolve_relations()


def test_relation_evaluation(settings_with_relations):
    settings_with_relations.resolve_relations()
    assert settings_with_relations.a.value == 6


def test_equation_with_self():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.a.relation = "a+2"
    settings.resolve_relations()
    assert settings.a.value == 3


def test_equation_with_self_and_another():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.b = None
    settings.a.relation = "a+2"
    settings.b.relation = "a+3"
    settings.resolve_relations()
    assert settings.a.value == 3
    assert settings.b.value == 6


def test_build_relation_hierarchy_with_nested_equations():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.b = 2
    settings.c = 1
    eq1 = pyqsl.Equation(equation="a+1")
    eq2 = pyqsl.Equation(equation="b+p1", parameters={"p1": eq1})
    settings.c.relation = eq2
    graph = settings.get_relation_hierarchy()
    assert set(graph.nodes) == set(["a", "b", "c"])


def test_nested_equations():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.b = 2
    settings.c = 1
    eq1 = pyqsl.Equation(equation="a+1")
    eq2 = pyqsl.Equation(equation="b+p1", parameters={"p1": eq1})
    settings.c.relation = eq2
    settings.resolve_relations()
    assert settings.c.value == 4


def test_nested_equations_with_cycle():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.b = 2
    settings.c = 1
    eq1 = pyqsl.Equation(equation="a+1")
    eq2 = pyqsl.Equation(equation="b+p1", parameters={"p1": eq1})
    settings.c.relation = eq2
    settings.b.relation = "c"
    with pytest.raises(ValueError):
        settings.resolve_relations()


def test_lookup_add_missing_parameters(settings):
    settings.amplitude.relation = pyqsl.LookupTable(
        coordinates={"frequency": [0, 1, 2]}, data=[0.1, 0.2, 0.3]
    )
    assert set(settings.amplitude.relation.parameters.keys()) == set(["frequency"])


def test_lookup_raise_if_inconsistent_shape(settings):
    with pytest.raises(ValueError):
        settings.amplitude.relation = pyqsl.LookupTable(
            coordinates={"frequency": [0, 1, 2]}, data=[0.1, 0.2]
        )
    with pytest.raises(ValueError):
        settings.amplitude.relation = pyqsl.LookupTable(
            coordinates={"frequency": [0, 1, 2], "amplitude": [1, 2]},
            data=[0.1, 0.2, 0.3],
        )


def test_lookup_evaluation(settings):
    settings.amplitude.relation = pyqsl.LookupTable(
        coordinates={"frequency": [0, 1, 2]}, data=[0.1, 0.2, 0.3]
    )
    settings.frequency.value = 1
    settings.resolve_relations()
    assert settings.amplitude.value == 0.2
    settings.frequency.value = 0.5
    settings.resolve_relations()
    assert pytest.approx(settings.amplitude.value) == 0.15


def test_lookup_evaluation_2d(settings):
    settings.amplitude.relation = pyqsl.LookupTable(
        coordinates={"frequency": [0, 1], "amplitude": [1, 2, 3]},
        data=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    )
    settings.amplitude.value = 2
    settings.frequency.value = 0.5
    settings.resolve_relations()
    assert pytest.approx(settings.amplitude.value) == 0.35


def test_nested_lookup_and_equation(settings):
    lut = pyqsl.LookupTable(data=[0.1, 0.2, 0.3], coordinates={"frequency": [1, 2, 3]})
    eq = pyqsl.Equation(equation="amplitude + lut", parameters={"lut": lut})
    settings.frequency.value = 2
    settings.amplitude.value = 1
    settings.amplitude.relation = eq
    settings.resolve_relations()
    assert pytest.approx(settings.amplitude.value) == 1.2


def test_lookup_table_options(settings):
    settings.amplitude.relation = pyqsl.LookupTable(
        coordinates={"frequency": [0, 1, 2]}, data=[0.1, 0.2, 0.3]
    )
    settings.frequency = -1
    with pytest.raises(RelationEvaluationError):
        settings.resolve_relations()

    fill_value = -0.1
    settings.amplitude.relation.interpolation_options = {'fill_value': fill_value, "bounds_error": False}
    settings.resolve_relations()
    assert settings.amplitude.value == fill_value
        
    

def test_nodes_with_relation_is_correct():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.b = 2
    settings.c = None
    settings.c.relation = pyqsl.Equation(equation="a+b")
    nodes_with_relation = settings.resolve_relations()
    assert settings.c.name in nodes_with_relation
    assert settings.a.name not in nodes_with_relation
    assert settings.b.name not in nodes_with_relation


def test_chained_relations_with_inactive_relation():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.b = 2
    settings.c = None
    settings.d = 3
    settings.f = None
    settings.c.relation = pyqsl.Equation(equation="a + b")
    settings.d.relation = pyqsl.Equation(equation="c")
    settings.d.use_relation = False
    settings.f.relation = pyqsl.Equation(equation="d")
    nodes_with_relation = settings.resolve_relations()
    assert "c" in nodes_with_relation
    assert "d" not in nodes_with_relation
    assert "f" in nodes_with_relation


def test_function():
    settings = pyqsl.Settings()
    settings.a = [0, 1]
    settings.b = None
    settings.b.relation = pyqsl.Function(function=np.mean, parameters={"a": "a"})
    parameters_with_relations = settings.resolve_relations()
    assert parameters_with_relations == ["b"]
    assert settings.b.value == pytest.approx(0.5)

    def my_own_function(b, c, d):
        return b + c + d

    settings.e = None
    settings.e.relation = pyqsl.Function(
        function=my_own_function, function_arguments={"c": 0.5, "d": 1.0}
    )
    parameters_with_relations = settings.resolve_relations()
    assert parameters_with_relations == ["b", "e"]
    assert settings.e.value == pytest.approx(2.0)


def test_errors_in_relations():
    settings = pyqsl.Settings()
    settings.a = 2
    settings.b = pyqsl.Setting(relation='ab')
    with pytest.raises(KeyError):
        settings.resolve_relations()
