import pyqsl
import numpy as np


def test_convert_sweeps_to_standard_form():
    a = pyqsl.Setting(name='a', value=2)
    sweeps = {a: np.linspace(0, 1, 3), 'b': [0, 1]}
    std_form = pyqsl.common.convert_sweeps_to_standard_form(sweeps)
    assert isinstance(list(sweeps.keys())[0], pyqsl.Setting)
    assert isinstance(list(std_form.keys())[0], str)
    assert isinstance(list(std_form.keys())[1], str)


def test_convert_sweeps_to_standard_form_for_shapes():
    a = pyqsl.Setting(name='a', value=2)
    sweeps = {a: [[0, 1], [2, 3, 4]]}
    std_form = pyqsl.common.convert_sweeps_to_standard_form(sweeps)
    assert len(std_form['a']) == 2
    assert len(std_form['a'][0]) == 2
    assert len(std_form['a'][1]) == 3

    sweeps = {a: [{'first', 'second'}, {'third', 'fourth'}]}
    std_form = pyqsl.common.convert_sweeps_to_standard_form(sweeps)
    assert len(std_form['a']) == 2
    assert len(std_form['a'][0]) == 2
    assert len(std_form['a'][1]) == 2
