import numpy as np

from scramjet_rl.physics.shock import shock_location_from_pressure, shock_on_lip_error


def test_shock_location_detects_pressure_jump():
    pressure = np.zeros((10, 20), dtype=np.float32)
    pressure[:, 12:] = 10.0
    x, y = shock_location_from_pressure(pressure)
    assert 0.45 < x < 0.7
    assert 0.0 <= y <= 1.0
    assert shock_on_lip_error(pressure) >= 0.0
