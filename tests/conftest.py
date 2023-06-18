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
