from dataclasses import dataclass

import healpy as hp
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pytz
import quivr as qv
from adam_core.coordinates import CartesianCoordinates, transform_coordinates
from adam_core.coordinates.origin import OriginCodes
from adam_core.observers import Observers
from adam_core.observers.utils import OBSERVATORY_PARALLAX_COEFFICIENTS
from adam_core.time import Timestamp
from astropy.time import Time


@dataclass
class Survey:
    # The observatory code
    observatory_code: str
    # The local observing start time (in 24 hour format)
    local_observing_start_time: int
    # The observing duration (in hours)
    observing_duration: int
    # The minimum number of visits to a field per night
    visits_per_night: int
    # The start night (UTC MJD)
    start_night: int
    # The end night (UTC MJD)
    end_night: int
    # The exposure time (in seconds)
    exposure_time: float
    # The slew time (in seconds)
    slew_time: float
    # Filters
    filters: list[str]


class SurveyPointings(qv.Table):

    exposure_id = qv.LargeStringColumn()
    exposure_start = Timestamp.as_column()
    exposure_duration = qv.Float64Column()
    filter = qv.LargeStringColumn()
    field_id = qv.Int64Column()
    field_ra = qv.Float64Column()
    field_dec = qv.Float64Column()
    five_sigma_depth = qv.Float64Column()
    observatory_code = qv.LargeStringColumn()
    observing_night = qv.Int64Column()


class SurveyFootprint(qv.Table):

    pixel_id = qv.Int64Column()
    ra = qv.Float64Column()
    dec = qv.Float64Column()
    x = qv.Float64Column()
    y = qv.Float64Column()
    z = qv.Float64Column()
    zenith_distance = qv.Float64Column(nullable=True)
    nside = qv.IntAttribute()

    @property
    def r(self) -> npt.NDArray[np.float64]:
        """
        Pointing vector to each pixel.
        """
        return np.array(self.table.select(["x", "y", "z"]))

    @classmethod
    def create(cls, nside: int = 8, nest: bool = True) -> "SurveyFootprint":
        """
        Create a SkyFootprint object for a given HEALPix resolution.

        Parameters
        ----------
        nside : int
            The HEALPix resolution parameter.
        nest : bool, optional
            Whether to use NEST indexing.
        """
        pixels = np.arange(hp.nside2npix(nside))
        ra, dec = hp.pix2ang(nside, pixels, nest=nest, lonlat=True)
        x, y, z = hp.pix2vec(nside, pixels, nest=nest)

        return cls.from_kwargs(
            pixel_id=pixels,
            ra=ra,
            dec=dec,
            x=x,
            y=y,
            z=z,
            nside=nside,
        )


def calculate_zenith_angle(
    survey_footprint: SurveyFootprint, observatory_code: str, time: Timestamp
) -> npt.NDArray[np.float64]:
    """
    Calculate the zenith angle for each field in the survey footprint at a given time.

    Parameters
    ----------
    survey_footprint : SurveyFootprint
        The survey footprint to calculate the zenith angle for.
    observatory_code : str
        The observatory code to calculate the zenith angle for.
    time : Timestamp
        The time to calculate the zenith angle for.

    Returns
    -------
    zenith_distances : np.ndarray
        The zenith angle for each field in the survey footprint.
    """

    observatory = Observers.from_code(observatory_code, time)
    obs_coords = transform_coordinates(
        observatory.coordinates,
        frame_out="equatorial",
        origin_out=OriginCodes.EARTH,
        representation_out=CartesianCoordinates,
    )
    zenith_vector = obs_coords.r_hat[0]
    cos_zenith = np.dot(survey_footprint.r, zenith_vector)
    zenith_distances = np.degrees(np.arccos(cos_zenith))

    return zenith_distances


def sort_footprint_scanning_pattern(
    survey_footprint: SurveyFootprint,
) -> SurveyFootprint:
    """
    Sort survey footprint into RA-Dec scanning pattern, handling RA wraparound.

    Parameters
    ----------
    survey_footprint : SurveyFootprint
        The survey footprint to sort.

    Returns
    -------
    sorted_survey_footprint : SurveyFootprint
        The sorted survey footprint.
    """
    if len(survey_footprint) == 0:
        return survey_footprint

    ra = survey_footprint.ra.to_numpy()
    dec = survey_footprint.dec.to_numpy()

    # Handle RA wraparound by finding largest gap
    ra_range = np.max(ra) - np.min(ra)
    if ra_range > 180:
        ra_sorted = np.sort(ra)
        gaps = np.diff(ra_sorted)
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)
            ra_start = ra_sorted[max_gap_idx + 1]
            ra_continuous = np.where(ra < ra_start, ra + 360.0, ra)
        else:
            ra_continuous = ra
    else:
        ra_continuous = ra

    # Sort by RA then Dec
    sort_indices = np.lexsort((dec, ra_continuous))
    return survey_footprint.take(sort_indices)


def select_footprint_for_night(
    survey_footprint: SurveyFootprint,
    observatory_code: str,
    observation_times: Timestamp,
    fields_per_night: int,
    visits_per_night: int,
    max_zenith_angle: float,
):
    """
    Given the survey footprint, the observation times for this night and the number of visits that need to be made
    to each field within the night, select the fields that need to be observed for this night.T

    We find the fields with lowest zenith angles over the course of the night.

    We do this by calculating the zenith angle at the start of each scan and collecting the fields that are visible at that time. We then select the n_fields
    with the lowest combined zenith angles (simple sum at each scan).

    Parameters
    ----------
    survey_footprint : SurveyFootprint
        Survey footprint.
    observatory_code : str
        Observatory code.
    observation_times : Timestamp
        The observation times for this night.
    fields_per_night : int
        The number of unique fields to select for each night. Each field should be visited visits_per_night times.
        For example, for the construction of tracklets.
    visits_per_night : int
        The number of visits to select for each night.
    max_zenith_angle : float
        The maximum zenith angle accessible to the observatory.

    Returns
    -------
    selected_fields : SurveyFootprint
        The survey footprint with the fields selected for this night.
    """
    assert len(observation_times) % fields_per_night == 0

    # Zenith angles for each field
    zenith_angles = np.empty((len(survey_footprint), visits_per_night))
    for scan in range(visits_per_night):

        times = observation_times[
            scan * fields_per_night : (scan + 1) * fields_per_night
        ]
        zenith_angles_i = calculate_zenith_angle(
            survey_footprint, observatory_code, times[0]
        )
        zenith_angles[:, scan] = np.where(
            zenith_angles_i < max_zenith_angle, zenith_angles_i, np.inf
        )

    # Select the fields with the lowest combined zenith angles
    selected_fields_indices = np.argsort(np.sum(zenith_angles, axis=1))
    selected_fields = survey_footprint.take(selected_fields_indices)[:fields_per_night]

    return selected_fields


def create_survey_pointings(
    surveys: list[Survey], survey_footprint: SurveyFootprint
) -> SurveyPointings:
    """
    Create a pointing schedule for the given surveys matched to the survey footprint.

    Parameters
    ----------
    surveys: list[Survey]
        The surveys to create pointings for.
    survey_footprint: SurveyFootprint
        The footprint to create pointings for.

    Returns
    -------
    SurveyPointings
        The survey pointings.
    """
    survey_pointings = SurveyPointings.empty()

    for survey in surveys:

        # Nights on which observations will be made (UTC)
        observing_nights = np.arange(survey.start_night, survey.end_night, 1)

        # Number of exposures to be made in each night (observing_duration / (exposure_time + slew_time))
        exposure_times_within_night = (
            np.arange(
                0 * 60 * 60,
                survey.observing_duration * 60 * 60,
                survey.exposure_time + survey.slew_time,
            )
            / 86400
        )

        # The number of exposures to be made in each night
        num_exposures = len(exposure_times_within_night)

        # The number of unique fields to be observed in each night
        num_fields_per_night = np.floor(num_exposures / survey.visits_per_night).astype(
            int
        )
        exposure_times_within_night = exposure_times_within_night[
            : survey.visits_per_night * num_fields_per_night
        ]
        num_exposures = len(exposure_times_within_night)

        # Get the timezone of the current observatory
        timezone = pytz.timezone(
            OBSERVATORY_PARALLAX_COEFFICIENTS.select(
                "code", survey.observatory_code
            ).timezone()[0]
        )

        # Calculate the mean five sigma depth for each night
        five_sigma_depth_night = np.random.uniform(22, 25, len(observing_nights))  # mag
        five_sigma_depth_std = np.random.uniform(0.1, 0.5, len(observing_nights))  # mag

        current_filter_index = 0
        survey_pointings = SurveyPointings.empty()
        for i, night in enumerate(observing_nights):

            # Calculate the mean five sigma depth for the current night
            five_sigma_depth = np.random.normal(
                five_sigma_depth_night[i], five_sigma_depth_std[i], num_exposures
            )

            # Compute the offset from UTC for the current night
            utc_offset = (
                Time(night, format="mjd", scale="utc")
                .datetime.astimezone(timezone)
                .utcoffset()
                .total_seconds()
                / 86400
            )

            # Compute the observation times for the current night (in UTC)
            observation_times_night = (
                night
                - (24 - survey.local_observing_start_time) / 24
                + exposure_times_within_night
                + utc_offset
            )

            # Compute the visible footprint at median observation time
            observation_times_night = Timestamp.from_mjd(
                np.array(observation_times_night), scale="utc"
            )

            selected_footprint = select_footprint_for_night(
                survey_footprint,
                survey.observatory_code,
                observation_times_night,
                num_fields_per_night,
                survey.visits_per_night,
                70.0,
            )
            selected_footprint = sort_footprint_scanning_pattern(selected_footprint)

            # Create the survey pointings for the current night
            field_ids = np.hstack(
                [
                    selected_footprint.pixel_id.to_pylist()
                    for _ in range(survey.visits_per_night)
                ]
            )
            field_ra = np.hstack(
                [
                    selected_footprint.ra.to_pylist()
                    for _ in range(survey.visits_per_night)
                ]
            )
            field_dec = np.hstack(
                [
                    selected_footprint.dec.to_pylist()
                    for _ in range(survey.visits_per_night)
                ]
            )

            exposure_ids = []
            filters = []
            for scan in range(survey.visits_per_night):
                current_filter_index += 1
                if current_filter_index >= len(survey.filters):
                    current_filter_index = 0

                exposure_ids += [
                    f"{survey.observatory_code}_{night}_{pixel_id:06d}_{scan:02d}"
                    for pixel_id in selected_footprint.pixel_id.to_pylist()
                ]
                filters += [
                    survey.filters[current_filter_index]
                    for _ in selected_footprint.pixel_id.to_pylist()
                ]

            survey_pointings_i = SurveyPointings.from_kwargs(
                exposure_id=exposure_ids,
                exposure_start=observation_times_night,
                exposure_duration=pa.repeat(survey.exposure_time, len(exposure_ids)),
                filter=filters,
                field_id=field_ids,
                field_ra=field_ra,
                field_dec=field_dec,
                five_sigma_depth=five_sigma_depth,
                observatory_code=pa.repeat(survey.observatory_code, len(exposure_ids)),
                observing_night=pa.repeat(night, len(exposure_ids)),
            )

            survey_pointings = qv.concatenate([survey_pointings, survey_pointings_i])

    return survey_pointings.sort_by(["exposure_start.days", "exposure_start.nanos"])
