import multiprocessing as mp
import os
import uuid
import warnings
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import quivr as qv
import ray
from adam_core.propagator.utils import _iterate_chunk_indices
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp
from jax import Array
from quivr.validators import and_, ge, le
from scipy.stats import skewnorm

from .observatory import Observatory
from .pointings import Pointings


class NoiseCatalog(qv.Table):

    id = qv.LargeStringColumn(default=lambda: uuid.uuid4().hex)
    exposure_id = qv.LargeStringColumn(nullable=True)
    time = Timestamp.as_column()
    ra = qv.Float64Column(validator=and_(ge(0), le(360)))
    dec = qv.Float64Column(validator=and_(ge(-90), le(90)))
    ra_sigma = qv.Float64Column(nullable=True)
    dec_sigma = qv.Float64Column(nullable=True)
    mag = qv.Float64Column(nullable=True)
    mag_sigma = qv.Float64Column(nullable=True)
    observatory_code = qv.LargeStringColumn()
    filter = qv.LargeStringColumn(nullable=True)
    exposure_start_time = Timestamp.as_column(nullable=True)
    exposure_duration = qv.Float64Column(nullable=True)
    exposure_seeing = qv.Float64Column(nullable=True)
    exposure_depth_5sigma = qv.Float64Column(nullable=True)
    object_id = qv.LargeStringColumn(nullable=True)
    catalog_id = qv.LargeStringColumn()


def magnitude_model(
    n: int,
    depth: float,
    scale: float,
    skewness: float,
    brightness_limit: Optional[float] = None,
    seed: Optional[int] = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
    ValueError : When the magnitudes cannot be sampled above the brightness limit after 25 attempts.
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
            if i == 25:
                raise ValueError(
                    "Could not sample magnitudes above the brightness limit after 25 attempts."
                )

    mag_err = rng.uniform(0.01, 0.3, n)
    return mag, mag_err


@jax.jit
def identify_within_circle(
    ra: npt.NDArray[np.float64],
    dec: npt.NDArray[np.float64],
    center_ra: float,
    center_dec: float,
    radius: float,
) -> Array:
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
    mask : jnp.ndarray
        A boolean mask that selects the points within the circle.
    """
    # Convert the coordinates to Cartesian (on a unit sphere)
    rad_ra = jnp.radians(ra)
    rad_dec = jnp.radians(dec)
    x = jnp.cos(rad_ra) * jnp.cos(rad_dec)
    y = jnp.sin(rad_ra) * jnp.cos(rad_dec)
    z = jnp.sin(rad_dec)

    # Convert the center to Cartesian
    rad_center_ra = jnp.radians(center_ra)
    rad_center_dec = jnp.radians(center_dec)
    center_x = jnp.cos(rad_center_ra) * jnp.cos(rad_center_dec)
    center_y = jnp.sin(rad_center_ra) * jnp.cos(rad_center_dec)
    center_z = jnp.sin(rad_center_dec)

    # Calculate the angle between the points and the center
    # The cosine of the angle is equal to the dot product of the two vectors
    cos_angle = x * center_x + y * center_y + z * center_z

    return cos_angle >= jnp.cos(jnp.radians(radius))


def add_noise(
    pointings: Pointings,
    observatory: Observatory,
    density: float,
    tag: Optional[str] = "noise",
    seed: Optional[int] = None,
) -> NoiseCatalog:
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
    tag : str, optional
        The tag to add to the noise detections. This is stored in the `catalog_id` column.
    seed : int, optional
        The seed for the random number generator.

    Returns
    -------
    noise : NoiseCatalog
        The catalog of noise detections.
    """
    if observatory.fov.camera_model != "circle":
        raise ValueError("Only circular camera model are supported at the moment")

    bright_limits = {
        f: observatory.bright_limit[i] for i, f in enumerate(observatory.filters)
    }

    noise = NoiseCatalog.empty()
    n_pointings = len(pointings)

    # Bounds derived from looking exposures in the NOIRLab Source Catalog
    rng = np.random.default_rng(seed=seed)
    mag_scale = rng.uniform(1.5, 3.0, n_pointings)
    mag_skewness = rng.uniform(-25, -5, n_pointings)

    radius = observatory.fov.circle_radius
    n_dets = int(4 * np.pi * (180**2 / np.pi**2) * density)

    for i, pointing in enumerate(pointings):

        ra = pointing.fieldRA_deg[0].as_py()
        dec = pointing.fieldDec_deg[0].as_py()
        filter_i = pointing.filter[0].as_py()

        # Generate random detections on the whole sky
        # as unit filter
        rng = np.random.default_rng(seed=seed)
        u = rng.random(n_dets)
        v = rng.random(n_dets)

        ra_dets = 2 * np.pi * u
        dec_dets = np.arccos(2 * v - 1) - np.pi / 2

        ra_dets = np.degrees(ra_dets)
        dec_dets = np.degrees(dec_dets)

        # Calculate the magnitude and magnitude errors
        try:
            mag, mag_err = magnitude_model(
                n_dets,
                pointing.fieldFiveSigmaDepth_mag[0].as_py(),
                mag_scale[i],
                mag_skewness[i],
                brightness_limit=bright_limits[filter_i],
            )
        except ValueError as e:
            warnings.warn(
                f"Skipping pointing {pointing.observationId[0].as_py()}: {e}",
                UserWarning,
            )
            continue

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

        num_obs = len(ra_dets)
        exposure_id = pa.repeat(pointing.observationId[0], num_obs)
        observation_time = pa.repeat(pointing.exposure_midpoint()[0], num_obs)
        observatory_code = pa.repeat(observatory.code, num_obs)
        filter = pa.repeat(filter_i, num_obs)
        exposure_start_time = Timestamp.from_mjd(
            pa.repeat(pointing.observationStartMJD_TAI[0], num_obs),
            scale="tai",
        )
        exposure_duration = pa.repeat(pointing.visitExposureTime[0], num_obs)
        exposure_seeing = pa.repeat(pointing.seeingFwhmEff_arcsec[0], num_obs)
        exposure_depth_5sigma = pa.repeat(pointing.fieldFiveSigmaDepth_mag[0], num_obs)
        catalog_id = pa.repeat(tag, num_obs)

        noise_pointing = NoiseCatalog.from_kwargs(
            exposure_id=exposure_id,
            time=Timestamp.from_mjd(observation_time, scale="tai"),
            ra=ra_dets,
            dec=dec_dets,
            ra_sigma=astrometric_error_arcsec,
            dec_sigma=astrometric_error_arcsec,
            mag=mag,
            mag_sigma=mag_err,
            observatory_code=observatory_code,
            filter=filter,
            exposure_start_time=exposure_start_time,
            exposure_duration=exposure_duration,
            exposure_seeing=exposure_seeing,
            exposure_depth_5sigma=exposure_depth_5sigma,
            object_id=None,
            catalog_id=catalog_id,
        )

        noise = qv.concatenate([noise, noise_pointing])

    return noise


def noise_worker(
    out_dir: str,
    pointing_ids: pa.Array,
    pointing_ids_indices: tuple[int, int],
    pointings: Pointings,
    observatory: Observatory,
    density: float,
    tag: Optional[str] = "noise",
    seed: Optional[int] = None,
) -> str:
    """
    Generate noise for a subset of the pointings.

    Parameters
    ----------
    out_dir : str
        The output directory where to save the noise detections.
    pointing_ids : pa.Array
        The observation IDs of the pointings.
    pointing_ids_indices : tuple[int, int]
        The indices of the subset of pointings to process.
    pointings : Pointings
        The pointings to generate noise detections for.
    observatory : Observatory
        The observatory that took the pointings.
    density : float
        The density of detections per square degree.
    tag : str, optional
        The tag to add to the noise detections. This is stored in the `catalog_id` column.
    seed : int, optional
        The seed for the random number generator.

    Returns
    -------
    noise_file : str
        The path to the saved noise detections.
    """
    pointing_ids = pointing_ids[pointing_ids_indices[0] : pointing_ids_indices[1]]
    pointings_chunk = pointings.apply_mask(
        pc.is_in(
            pointings.observationId,
            pointing_ids,
        )
    )

    noise = add_noise(
        pointings_chunk,
        observatory,
        density,
        seed=seed,
    )

    noise_file = os.path.join(
        out_dir,
        f"noise_{pointing_ids_indices[0]:08d}_{pointing_ids_indices[1]:08d}.parquet",
    )
    noise.to_parquet(noise_file)
    return noise_file


noise_worker_remote = ray.remote(noise_worker)
noise_worker_remote.options(num_cpus=1)


def generate_noise(
    output_dir: str,
    pointings: Pointings,
    observatory: Observatory,
    density: float,
    tag: Optional[str] = "noise",
    seed: Optional[int] = None,
    chunk_size: int = 100,
    max_processes: Optional[int] = 1,
    cleanup: bool = True,
) -> str:
    """
    Generate noise detections for each pointing in pointings using the given observatory and density.

    Parameters
    ----------
    pointings : Pointings
        The pointings to generate noise detections for.
    observatory : Observatory
        The observatory that took the pointings.
    density : float
        The density of detections per square degree.
    tag : str, optional
        The tag to add to the noise detections. This is stored in the `catalog_id` column.
    seed : int, optional
        The seed for the random number generator.
    chunk_size : int, optional
        The size of the chunks to process.
    max_processes : int, optional
        The maximum number of processes to use. If None, all available CPUs will be used.
    cleanup : bool, optional
        Whether to delete the temporary chunked files after concatenating them
        into the final `noise_file`.

    Returns
    -------
    noise_file : str
        The path to the saved noise detections.
    """
    if max_processes is None:
        max_processes = mp.cpu_count()

    pointing_ids = pointings.observationId
    noise_catalog = NoiseCatalog.empty()

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    noise_file = os.path.join(output_dir, f"noise_{density:.3f}.parquet")
    noise_catalog.to_parquet(noise_file)
    noise_file_writer = pq.ParquetWriter(noise_file, NoiseCatalog.schema)

    use_ray = initialize_use_ray(max_processes)
    if use_ray:

        pointing_ids_ref = ray.put(pointing_ids)
        pointings_ref = ray.put(pointings)
        observatory_ref = ray.put(observatory)

        chunk_size = min(int((len(pointings) / max_processes)), chunk_size)
        chunk_size = max(1, int(chunk_size))

        futures = []
        for pointing_ids_indices in _iterate_chunk_indices(pointing_ids, chunk_size):
            futures.append(
                noise_worker_remote.remote(
                    output_dir,
                    pointing_ids_ref,
                    pointing_ids_indices,
                    pointings_ref,
                    observatory_ref,
                    density,
                    tag=tag,
                    seed=seed,
                )
            )

            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)

                noise_catalog_chunk_file = ray.get(finished[0])
                noise_catalog_chunk = NoiseCatalog.from_parquet(
                    noise_catalog_chunk_file
                )
                noise_file_writer.write_table(noise_catalog_chunk.table)

                if cleanup:
                    os.remove(noise_catalog_chunk_file)

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)

            noise_catalog_chunk_file = ray.get(finished[0])
            noise_catalog_chunk = NoiseCatalog.from_parquet(noise_catalog_chunk_file)
            noise_file_writer.write_table(noise_catalog_chunk.table)

            if cleanup:
                os.remove(noise_catalog_chunk_file)

    else:

        for pointing_ids_indices in _iterate_chunk_indices(pointing_ids, chunk_size):
            noise_catalog_chunk_file = noise_worker(
                output_dir,
                pointing_ids,
                pointing_ids_indices,
                pointings,
                observatory,
                density,
                tag=tag,
                seed=seed,
            )

            noise_catalog_chunk = NoiseCatalog.from_parquet(noise_catalog_chunk_file)
            noise_file_writer.write_table(noise_catalog_chunk.table)

            if cleanup:
                os.remove(noise_catalog_chunk_file)

    return noise_file
