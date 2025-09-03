from dataclasses import dataclass

import healpy as hp
import numpy as np
import numpy.typing as npt
import quivr as qv
from adam_core.time import Timestamp


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
        vectors = np.vstack((x, y, z)).T

        return cls.from_kwargs(
            pixel_id=pixels,
            ra=ra,
            dec=dec,
            x=x,
            y=y,
            z=z,
            nside=nside,
        )
