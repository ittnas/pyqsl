import pytest
import pyqsl


@pytest.fixture
def settings():
    settings = pyqsl.Settings()
    settings.amplitude = 4
    settings.amplitude.unit = 'V'
    settings.frequency = 2
    settings.frequency.unit = 'Hz'
    return settings


@pytest.fixture
def settings_with_relations():
    settings = pyqsl.Settings()
    settings.a = 1
    settings.b = 2
    settings.c = 3
    settings.d = 4
    settings.e = 5
    settings.a.relation = 'b + c'
    settings.b.relation = 'c'
    settings.e.relation = 'a'
    return settings
