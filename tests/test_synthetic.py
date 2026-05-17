from scramjet_rl.data.synthetic import SyntheticConfig, generate_synthetic_fields


def test_synthetic_shapes():
    arrays = generate_synthetic_fields(
        SyntheticConfig(
            num_samples=4,
            height=16,
            width=32,
            seed=1,
            mach_range=(4.0, 8.0),
            altitude_range_m=(10000.0, 30000.0),
            ramp_angle_range_deg=(4.0, 18.0),
        )
    )
    assert arrays["inputs"].shape == (4, 4)
    assert arrays["pressure"].shape == (4, 16, 32)
    assert arrays["temperature"].shape == (4, 16, 32)
    assert arrays["metrics"].shape == (4, 4)
