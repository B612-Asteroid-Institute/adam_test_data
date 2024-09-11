import numpy as np

from ..noise import fix_wrap_around, magnitude_model


def test_magnitude_model():
    # Test that magnitude_model returns the expected number of magnitudes and errors
    seed = 42
    n = 100
    skewness = 0.5
    scale = 0.1

    mag, mag_err = magnitude_model(n, 25, scale, skewness, seed)
    assert len(mag) == n
    assert len(mag_err) == n
    assert np.all(mag_err >= 0.01) and np.all(mag_err <= 0.3)


def test_fix_wrap_around():
    # Test that fix_wrap_around correctly adjusts RAs that are greater than 360 degrees or less than 0 degrees
    ra = np.array([0, 360, 361, -1, -2])
    ra_expected = np.array([0, 0, 1, 359, 358])
    np.testing.assert_array_equal(fix_wrap_around(ra), ra_expected)
