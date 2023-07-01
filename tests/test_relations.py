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
    assert equation.equation == 'x'
    setting = pyqsl.Setting('b')
    equation = pyqsl.relation.Equation(equation='a+1', parameters={'a': setting})
    assert equation.parameters == {'a': 'b'}


def test_paremeter_not_found_error():
    equation = pyqsl.relation.Equation(equation='a+1')
    assert equation.parameters == {'a': 'a'}


def test_equation_evaluation():
    equation = pyqsl.relation.Equation(equation='a+1', parameters={'a': 'b'})
    assert equation.evaluate({'a': 20}) == 21
