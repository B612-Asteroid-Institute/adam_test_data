import numpy as np
import pytest

from ..noise import identify_within_circle, magnitude_model


def test_magnitude_model() -> None:
    # Test that magnitude_model returns the expected number of magnitudes and errors
    seed = 42
    n = 100
    skewness = -11
    scale = 2.22
    depth = 25

    mag, mag_err = magnitude_model(n, depth, scale, skewness, seed=seed)
    assert len(mag) == n
    assert len(mag_err) == n
    assert np.all(mag_err >= 0.01) and np.all(mag_err <= 0.3)


def test_magnitude_model_brightness_limit() -> None:
    # Test that magnitude_model returns the expected number of magnitudes and errors
    seed = 42
    n = 100
    skewness = -11
    scale = 2.22
    depth = 25
    brightness_limit = 21

    mag, mag_err = magnitude_model(
        n, depth, scale, skewness, brightness_limit=brightness_limit, seed=seed
    )
    assert len(mag) == n
    assert len(mag_err) == n
    assert np.all(mag >= brightness_limit)


def test_magnitude_model_brightness_limit_raises() -> None:
    # Test that magnitude_model returns the expected number of magnitudes and errors
    seed = 42
    n = 100
    skewness = -11
    scale = 2.22
    depth = 25
    brightness_limit = 30

    with pytest.raises(
        ValueError,
        match="Could not sample magnitudes above the brightness limit after 25 attempts.",
    ):
        mag, mag_err = magnitude_model(
            n, depth, scale, skewness, brightness_limit=brightness_limit, seed=seed
        )


def test_identify_within_circle() -> None:
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
