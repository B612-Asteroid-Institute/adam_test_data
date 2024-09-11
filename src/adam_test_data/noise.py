from typing import Optional

import numpy as np
import pyarrow as pa
import quivr as qv
from scipy.stats import skewnorm

from .observatory import Observatory
from .pointings import Pointings


class Noise(qv.Table):

    FieldID = qv.LargeStringColumn()
    RA_deg = qv.Float64Column()
    Dec_deg = qv.Float64Column()
    astrometricSigma_deg = qv.Float64Column()
    PSFMag = qv.Float64Column()
    PSFMagSigma = qv.Float64Column()


def magnitude_model(
    n: int,
    depth: float,
    scale: float,
    skewness: float,
    brightness_limit: Optional[float] = None,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the distribution of magnitudes as a skewed normal distribution
    with mean parameter equal to the five-sigma depth. This was modeled after looking at exposures
    from the NOIRLab Source Catalog.

    Paremeters
    ----------
    n : int
        The number of magnitudes to simulate.
    depth : float
        The 5-sigma depth of the pointing.
    scale : float
        The scale parameter of the skewed normal distribution.
    skewness : float
        The skewness parameter of the skewed normal distribution.
    brightness_limit : float, optional
        The brightness limit of the pointing. Any magnitudes sampled below this limit will be
        resampled until they are above the limit.
    seed : int
        The seed for the random number generator.

    Returns
    -------
    mag : np.ndarray
        The simulated magnitudes.
    mag_err : np.ndarray
        The simulated magnitude errors.

    Raises
    ------
    ValueError : When the magnitudes cannot be sampled above the brightness limit after 10 attempts.
    """
    rng = np.random.default_rng(seed)
    skewnorm_random = skewnorm
    skewnorm_random.random_state = rng
    mag = skewnorm_random.rvs(a=skewness, loc=depth, scale=scale, size=n)
    if brightness_limit is not None:
        i = 0
        while np.any(mag < brightness_limit):
            mask = mag < brightness_limit
            mag[mask] = skewnorm_random.rvs(
                a=skewness, loc=depth, scale=scale, size=mask.sum()
            )

            i += 1
            if i == 10:
                raise ValueError(
                    "Could not sample magnitudes above the brightness limit after 10 attempts."
                )

    mag_err = rng.uniform(0.01, 0.3, n)
    return mag, mag_err


def identify_within_circle(
    ra: np.ndarray, dec: np.ndarray, center_ra: float, center_dec: float, radius: float
) -> np.ndarray:
    """
    Identify the points that are within a circle of radius `radius` centered at `center_ra`, `center_dec`.

    Parameters
    ----------
    ra : np.ndarray
        The right ascension of the points.
    dec : np.ndarray
        The declination of the points.
    center_ra : float
        The right ascension of the center of the circle.
    center_dec : float
        The declination of the center of the circle.
    radius : float
        The radius of the circle.

    Returns
    -------
    mask : np.ndarray
        A boolean mask that selects the points within the circle.
    """
    # Convert the coordinates to Cartesian (on a unit sphere)
    rad_ra = np.radians(ra)
    rad_dec = np.radians(dec)
    x = np.cos(rad_ra) * np.cos(rad_dec)
    y = np.sin(rad_ra) * np.cos(rad_dec)
    z = np.sin(rad_dec)

    # Convert the center to Cartesian
    rad_center_ra = np.radians(center_ra)
    rad_center_dec = np.radians(center_dec)
    center_x = np.cos(rad_center_ra) * np.cos(rad_center_dec)
    center_y = np.sin(rad_center_ra) * np.cos(rad_center_dec)
    center_z = np.sin(rad_center_dec)

    # Calculate the angle between the points and the center
    # The cosine of the angle is equal to the dot product of the two vectors
    angle = np.arccos(x * center_x + y * center_y + z * center_z)

    return angle <= np.radians(radius)


def fix_wrap_around(ra: np.ndarray) -> np.ndarray:
    """
    Adjust any RAs that are greater than 360 degrees or less than 0 degrees.

    Parameters
    ----------
    ra : np.ndarray
        The right ascensions to adjust.

    Returns
    -------
    ra : np.ndarray
        The adjusted right ascensions.
    """
    ra = np.where(ra >= 360, ra - 360, ra)
    ra = np.where(ra < 0, ra + 360, ra)
    return ra


def add_noise(
    pointings: Pointings,
    observatory: Observatory,
    density: float,
    seed: Optional[int] = None,
) -> Noise:
    """
    For each pointing in pointings, create noise observations with a fixed density per
    square degree. The magnitude distribution is modeled as a skewed normal distribution
    with the mean parameter equal to the five-sigma depth of the pointing. The scale and skewness
    are drawn from a uniform distribution. The astrometric error is drawn from a normal distribution
    with a standard deviation derived from the seeing FWHM of the pointing.

    Parameters
    ----------
    pointings : Pointings
        The pointings to add noise to.
    observatory : Observatory
        The observatory that took the pointings.
    density : float
        The density of detections per square degree.
    seed : int, optional
        The seed for the random number generator.

    Returns
    -------
    noise : Noise
        The noise observations.
    """
    if observatory.fov.camera_model != "circle":
        raise ValueError("Only circular camera model are supported at the moment")

    bright_limits = {
        f: observatory.bright_limit[i] for i, f in enumerate(observatory.filters)
    }

    noise = Noise.empty()
    n_pointings = len(pointings)

    # Bounds derived from looking exposures in the NOIRLab Source Catalog
    rng = np.random.default_rng(seed=seed)
    mag_scale = rng.uniform(1.5, 3.0, n_pointings)
    mag_skewness = rng.uniform(-25, -5, n_pointings)

    for i, pointing in enumerate(pointings):

        ra = pointing.fieldRA_deg[0].as_py()
        dec = pointing.fieldDec_deg[0].as_py()
        radius = observatory.fov.circle_radius

        # Calculate the box that contains the circle
        ra_min = ra - radius / np.cos(np.radians(dec))
        ra_max = ra + radius / np.cos(np.radians(dec))
        dec_min = dec - radius
        dec_max = dec + radius
        dra = ra_max - ra_min
        ddec = dec_max - dec_min

        if dec_max > 90 or dec_min < -90:
            # If we encounter this we will need to implement this
            # with Cartesian coordinates instead of spherical coordinates
            # Or implement great circle distance calculations in adam_core's
            # SphericalCoordinates class
            raise ValueError("The circular FOV is too close to the poles.")

        # Calculate the number of points to generate
        area = dra * ddec  # in square degrees
        n_dets = int(area * density)

        # Generate the detections
        ra_dets = np.random.uniform(ra_min, ra_max, n_dets)
        dec_dets = np.random.uniform(dec_min, dec_max, n_dets)

        # Calculate the magnitude and magnitude errors
        mag, mag_err = magnitude_model(
            n_dets,
            pointing.fieldFiveSigmaDepth_mag[0].as_py(),
            mag_scale[i],
            mag_skewness[i],
            brightness_limit=bright_limits[filter_i],
        )

        # Calculate the astrometric error (here we use the seeing FWHM of the pointing)
        fwhm = pointing.seeingFwhmEff_arcsec[0].as_py()
        astrometric_error_arcsec = np.abs(np.random.normal(0.0, fwhm / 2.355, n_dets))

        # Filter the detections that are within the circular FOV
        mask = identify_within_circle(ra_dets, dec_dets, ra, dec, radius)

        # Apply the mask
        ra_dets = ra_dets[mask]
        dec_dets = dec_dets[mask]
        mag = mag[mask]
        mag_err = mag_err[mask]
        astrometric_error_arcsec = astrometric_error_arcsec[mask]

        # Now lets fix any wrap-around issues
        ra_dets = fix_wrap_around(ra_dets)

        noise_pointing = Noise.from_kwargs(
            FieldID=pa.repeat(pointing.observationId[0].as_py(), len(ra_dets)),
            RA_deg=ra_dets,
            Dec_deg=dec_dets,
            astrometricSigma_deg=astrometric_error_arcsec / 3600,
            PSFMag=mag,
            PSFMagSigma=mag_err,
        )

        noise = qv.concatenate([noise, noise_pointing])

    return noise
