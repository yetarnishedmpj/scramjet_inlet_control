from scramjet_rl.rl.baseline import target_angle_for_mach


def test_target_angle_increases_with_mach():
    assert target_angle_for_mach(6.0) > target_angle_for_mach(4.0)
    assert 4.0 <= target_angle_for_mach(8.0) <= 18.0
