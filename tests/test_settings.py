import pyqsl
import pytest


def test_create_setting():
    with pytest.raises(TypeError):
        setting = pyqsl.Setting()

    empty_setting = pyqsl.Setting('frequency', unit='Hz')
    assert empty_setting.value is None
    setting = pyqsl.Setting('amplitude', 4)
    setting = pyqsl.Setting('amplitude', 4, unit='V')
    assert setting.value == 4
    assert setting.unit == 'V'
    assert setting.name == 'amplitude'


def test_create_settings():
    settings = pyqsl.Settings()
    settings.amplitude = 3
    assert settings.amplitude.value == 3
    assert settings.amplitude.name == 'amplitude'
    setting = pyqsl.Setting('amplitude')
    settings.amplitude = setting
    assert settings.amplitude.value is None
    settings.amplitude = 5
    assert settings.amplitude.value == 5


def test_overloading_add():
    settings = pyqsl.Settings()
    settings.amplitude = 5
    assert 3 + settings.amplitude == 8
    assert settings.amplitude + 3 == 8
    assert settings.amplitude + settings.amplitude == 10
    assert settings.amplitude + settings.amplitude != 9


def test_adding_setting_with_wrong_name():
    settings = pyqsl.Settings()
    settings.amplitude = 5
    setting = pyqsl.Setting('frequency', 4)
    with pytest.raises(ValueError):
        settings.amplitude = setting


def test_converting_to_dict(settings):
    settings_dict = settings.to_dict()
    print(settings_dict.keys())
    assert settings_dict['amplitude'] == 4


def test_adding_to_dict(settings):
    setting_dict = {settings.amplitude: [0, 1, 2]}
    assert setting_dict['amplitude'] == [0, 1, 2]
    assert setting_dict[settings.amplitude] == [0, 1, 2]
    setting_dict['amplitude'] = [0, 1]
    assert setting_dict[settings.amplitude] == [0, 1]
