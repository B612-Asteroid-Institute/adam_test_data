import numpy as np

from ..noise import fix_wrap_around, identify_within_circle, magnitude_model


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


def test_identify_within_circle():
    # Test that identify within circle can correctly handle a simple circle near the equator
    ra = np.array([30.0, 34.0, 32.0, 32.0, 32.0])
    dec = np.array([0.0, 0.0, 2.0, -2.0, 0.0])

    ra_center = 32.0
    dec_center = 0.0

    radius = 1.01
    mask = identify_within_circle(ra, dec, ra_center, dec_center, radius)
    expected_mask = np.array([False, False, False, False, True])
    np.testing.assert_array_equal(mask, expected_mask)

    radius = 2.01
    mask = identify_within_circle(ra, dec, ra_center, dec_center, radius)
    expected_mask = np.array([True, True, True, True, True])
    np.testing.assert_array_equal(mask, expected_mask)

    # Test wrap arounds past 360 degrees of RA
    ra = np.array([359.0, 1.0])
    dec = np.array([0.0, 0.0])
    ra_center = 0.0
    dec_center = 0.0

    radius = 1.01
    mask = identify_within_circle(ra, dec, ra_center, dec_center, radius)

    expected_mask = np.array([True, True])
    np.testing.assert_array_equal(mask, expected_mask)

    # Test wrap arounds near the poles
    # RA spans 0 to 360 here
    ra = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
    dec = np.array([89.0, 89.0, 89.0, 89.0, 89.0])
    ra_center = 0.0
    dec_center = 90.0

    radius = 1.01
    mask = identify_within_circle(ra, dec, ra_center, dec_center, radius)
    expected_mask = np.array([True, True, True, True, True])
    np.testing.assert_array_equal(mask, expected_mask)

    mask = identify_within_circle(ra, -dec, ra_center, dec_center, radius)
    expected_mask = np.array([False, False, False, False, False])
    np.testing.assert_array_equal(mask, expected_mask)
