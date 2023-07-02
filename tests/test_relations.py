import pyqsl
import pytest


def test_that_abstract_relation_cannot_be_created():
    with pytest.raises(TypeError):
        relation = pyqsl.relation.Relation()


def test_create_equation():
    equation = pyqsl.relation.Equation(equation='a+1', parameters={'a': 'b'})
    assert equation.parameters == {'a': 'b'}
    assert equation.equation == 'a+1'
    equation = pyqsl.relation.Equation()
    assert equation.parameters == {}
    assert equation.equation == '0'
    setting = pyqsl.Setting('b')
    equation = pyqsl.relation.Equation(equation='a+1', parameters={'a': setting})
    assert equation.parameters == {'a': 'b'}


def test_paremeter_not_found_error():
    equation = pyqsl.relation.Equation(equation='a+1')
    assert equation.parameters == {'a': 'a'}


def test_equation_evaluation():
    equation = pyqsl.relation.Equation(equation='a+1', parameters={'a': 'b'})
    assert equation.evaluate(a=20) == 21
    equation = pyqsl.relation.Equation(equation='a*2 + c')
    assert equation.evaluate(a=2, c=1) == 5


def test_build_relation_hierarchy(settings_with_relations):
    graph = settings_with_relations._build_relation_hierarchy()
    assert set(graph.nodes) == set(['a', 'b', 'c', 'e'])
    assert set(graph.edges) == {('a', 'e'), ('b', 'a'), ('c', 'a'), ('c', 'b')}


def test_that_self_reference_is_not_added(settings_with_relations):
    settings_with_relations.d.relation = 'd'
    graph = settings_with_relations._build_relation_hierarchy()
    assert set(graph.nodes) == set(['a', 'b', 'c', 'e', 'd'])
    assert set(graph.edges) == {('a', 'e'), ('b', 'a'), ('c', 'a'), ('c', 'b')}


def test_is_acyclic():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.b = 2
    settings.a.relation = 'b'
    settings.b.relation = 'a'
    graph = settings._build_relation_hierarchy()
    assert pyqsl.settings.is_acyclic(graph) is False
    settings.b.relation = None
    graph = settings._build_relation_hierarchy()
    assert pyqsl.settings.is_acyclic(graph) is True


def test_resolve_relations_with_cyclic_graph():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.b = 2
    settings.a.relation = 'b'
    settings.b.relation = 'a'
    with pytest.raises(ValueError):
        settings.resolve_relations()


def test_relation_evaluation(settings_with_relations):
    settings_with_relations.resolve_relations()
    assert settings_with_relations.a.value == 6


def test_equation_with_self():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.a.relation = 'a+2'
    settings.resolve_relations()
    assert settings.a.value == 3


def test_equation_with_self_and_another():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.b = None
    settings.a.relation = 'a+2'
    settings.b.relation = 'a+3'
    settings.resolve_relations()
    assert settings.a.value == 3
    assert settings.b.value == 6


def test_build_relation_hierarchy_with_nested_equations():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.b = 2
    settings.c = 1
    eq1 = pyqsl.Equation(equation='a+1')
    eq2 = pyqsl.Equation(equation='b+p1', parameters={'p1': eq1})
    settings.c.relation = eq2
    graph = settings._build_relation_hierarchy()
    assert set(graph.nodes) == set(['a', 'b', 'c'])


def test_nested_equations():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.b = 2
    settings.c = 1
    eq1 = pyqsl.Equation(equation='a+1')
    eq2 = pyqsl.Equation(equation='b+p1', parameters={'p1': eq1})
    settings.c.relation = eq2
    settings.resolve_relations()
    assert settings.c.value == 4


def test_nested_equations_with_cycle():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.b = 2
    settings.c = 1
    eq1 = pyqsl.Equation(equation='a+1')
    eq2 = pyqsl.Equation(equation='b+p1', parameters={'p1': eq1})
    settings.c.relation = eq2
    settings.b.relation = 'c'
    with pytest.raises(ValueError):
        settings.resolve_relations()


