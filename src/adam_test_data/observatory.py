from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class FieldOfView:
    camera_model: Literal["footprint", "circle"]
    footprint_path: Optional[str] = None
    fill_factor: Optional[float] = None
    circle_radius: Optional[float] = None
    footprint_edge_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        self.camera_model = self.camera_model

        if self.camera_model == "footprint":
            if self.footprint_path is None:
                raise ValueError(
                    "footprint_path is required for camera_model='footprint'"
                )
        elif self.camera_model == "circle":
            if self.fill_factor is None:
                raise ValueError("fill_factor is required for camera_model='circle'")
            if self.circle_radius is None:
                raise ValueError("circle_radius is required for camera_model='circle'")
            if self.footprint_edge_threshold is not None:
                raise ValueError(
                    "footprint_edge_threshold is only valid for camera_model='footprint'"
                )
        else:
            raise ValueError(f"Unknown camera model: {self.camera_model}")

    def to_string(self) -> str:
        string = "[FOV]\n"
        if self.camera_model == "footprint":
            string += f"camera_model = {self.camera_model}\nfootprint_path = {self.footprint_path}"
            if self.footprint_edge_threshold is not None:
                string += (
                    f"\nfootprint_edge_threshold = {self.footprint_edge_threshold}"
                )
        else:
            string += f"camera_model = {self.camera_model}\nfill_factor = {self.fill_factor}\ncircle_radius = {self.circle_radius}"

        return string


@dataclass
class Simulation:
    ang_fov: float
    fov_buffer: float
    picket: Optional[int] = 1
    healpix_order: Optional[int] = 6

    def to_string(self, observatory_code: str) -> str:
        return f"""[SIMULATION]
ar_ang_fov = {self.ang_fov}
ar_fov_buffer = {self.fov_buffer}
ar_picket = {self.picket}
ar_obs_code = {observatory_code}
ar_healpix_order = {self.healpix_order}"""


@dataclass
class Observatory:
    code: str
    filters: list[str]
    main_filter: str
    bright_limit: list[float]
    fov: FieldOfView
    simulation: Simulation


def observatory_to_sorcha_config(
    observatory: Observatory,
    time_range: Optional[list[float]] = None,
    randomization: bool = True,
    output_columns: Literal["basic", "all"] = "basic",
) -> str:
    """
    Create a Sorcha configuration file from an Observatory object and, optionally, a time range.

    Parameters
    ----------
    observatory : Observatory
        The observatory object to create the configuration file for.
    time_range : list[float], optional
        The time range to filter the pointings by, by default None.
    randomization : bool, optional
        Ramdomize the photometry and astrometry using the calculated uncertainties, by default True.
    output_columns : Literal["basic", "all"], optional
        The columns to output in the Sorcha output, by default "all".

    Returns
    -------
    str
        The Sorcha configuration file as a string.
    """
    sql_query = f"SELECT * FROM pointings WHERE observatory_code = '{observatory.code}'"
    if time_range is not None:
        sql_query += f" AND observationStartMJD >= {time_range[0]} AND observationStartMJD <= {time_range[1]}"

    # The main filter needs to be first in the list of filters
    if observatory.main_filter in observatory.filters:
        observatory.filters.remove(observatory.main_filter)
        observatory.filters.insert(0, observatory.main_filter)
    else:
        raise ValueError(
            f"Main filter {observatory.main_filter} not in list of filters"
        )

    config = f"""
# Sorcha Configuration File - ADAM Test Data - {observatory.code}

[INPUT]
ephemerides_type = ar
eph_format = csv
size_serial_chunk = 5000
aux_format = csv
pointing_sql_query = {sql_query}

{observatory.simulation.to_string(observatory.code)}

[FILTERS]
observing_filters = {','.join(observatory.filters)}

[SATURATION]
bright_limit = {','.join(map(str, observatory.bright_limit))}

[PHASECURVES]
phase_function = HG

{observatory.fov.to_string()}

[FADINGFUNCTION]
fading_function_on = True
fading_function_width = 0.1
fading_function_peak_efficiency = 1

[OUTPUT]
output_format = csv
output_columns = {output_columns}

[LIGHTCURVE]
lc_model = none

[ACTIVITY]
comet_activity = none

[EXPERT]
randomization_on = {randomization}
vignetting_on = True
trailing_losses_on = True
"""
    return config
