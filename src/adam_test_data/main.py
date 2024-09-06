import os
import sqlite3 as sql
import subprocess
from typing import Literal, Optional, Union

import quivr as qv

from .observatory import Observatory, observatory_to_sorcha_config
from .pointings import Pointings
from .populations import (
    SmallBodies,
    orbits_to_sorcha_dataframe,
    photometric_properties_to_sorcha_dataframe,
)


class SorchaOutputBasic(qv.Table):
    # TODO: It would be really nice if the basic outputs included the FieldID
    # so that we could match the observations to the pointings
    ObjID = qv.LargeStringColumn()
    fieldMJD_TAI = qv.Float64Column()
    fieldRA_deg = qv.Float64Column()
    fieldDec_deg = qv.Float64Column()
    RA_deg = qv.Float64Column()
    Dec_deg = qv.Float64Column()
    astrometricSigma_deg = qv.Float64Column()
    optFilter = qv.LargeStringColumn()
    trailedSourceMag = qv.Float64Column()
    trailedSourceMagSigma = qv.Float64Column()
    fiveSigmaDepth_mag = qv.Float64Column()
    phase_deg = qv.Float64Column()
    Range_LTC_km = qv.Float64Column()
    RangeRate_LTC_km_s = qv.Float64Column()
    Obj_Sun_LTC_km = qv.Float64Column()


class SorchaOutputAll(qv.Table):

    # Some of the commented-out columns can be retroactively
    # calculated by joining with inputs to sorcha.

    ObjID = qv.LargeStringColumn()
    FieldID = qv.LargeStringColumn()
    fieldMJD_TAI = qv.Float64Column()
    fieldJD_TDB = qv.Float64Column()
    Range_LTC_km = qv.Float64Column()
    RangeRate_LTC_km_s = qv.Float64Column()
    RATrue_deg = qv.Float64Column()
    RARateCosDec_deg_day = qv.Float64Column()
    DecTrue_deg = qv.Float64Column()
    DecRate_deg_day = qv.Float64Column()
    # Obj_Sun_x_LTC_km = qv.Float64Column()
    # Obj_Sun_y_LTC_km = qv.Float64Column()
    # Obj_Sun_z_LTC_km = qv.Float64Column()
    # Obj_Sun_vx_LTC_km_s = qv.Float64Column()
    # Obj_Sun_vy_LTC_km_s = qv.Float64Column()
    # Obj_Sun_vz_LTC_km_s = qv.Float64Column()
    # Obs_Sun_x_km = qv.Float64Column()
    # Obs_Sun_y_km = qv.Float64Column()
    # Obs_Sun_z_km = qv.Float64Column()
    # Obs_Sun_vx_km_s = qv.Float64Column()
    # Obs_Sun_vy_km_s = qv.Float64Column()
    # Obs_Sun_vz_km_s = qv.Float64Column()
    phase_deg = qv.Float64Column()
    # FORMAT = qv.LargeStringColumn()
    # x = qv.Float64Column()
    # y = qv.Float64Column()
    # z = qv.Float64Column()
    # xdot = qv.Float64Column()
    # ydot = qv.Float64Column()
    # zdot = qv.Float64Column()
    # epochMJD_TDB = qv.Float64Column()
    # H_filter = qv.Float64Column()
    # VR_r = qv.Float64Column()
    # GS = qv.Float64Column()
    # visitTime = qv.Float64Column()
    # visitExposureTime = qv.Float64Column()
    optFilter = qv.LargeStringColumn()
    # seeingFwhmGeom_arcsec = qv.Float64Column()
    # seeingFwhmEff_arcsec = qv.Float64Column()
    fieldFiveSigmaDepth_mag = qv.Float64Column()
    # fieldRA_deg = qv.Float64Column()
    # fieldDec_deg = qv.Float64Column()
    # rotSkyPos_deg = qv.Float64Column()
    # observatory_code = qv.LargeStringColumn()
    # H_r = qv.Float64Column()
    trailedSourceMagTrue = qv.Float64Column()
    PSFMagTrue = qv.Float64Column()
    fiveSigmaDepth_mag = qv.Float64Column()
    astrometricSigma_deg = qv.Float64Column()
    trailedSourceMagSigma = qv.Float64Column()
    SNR = qv.Float64Column()
    PSFMagSigma = qv.Float64Column()
    trailedSourceMag = qv.Float64Column()
    PSFMag = qv.Float64Column()
    RA_deg = qv.Float64Column()
    Dec_deg = qv.Float64Column()
    Obj_Sun_LTC_km = qv.Float64Column()


class SorchaOutputStats(qv.Table):

    ObjID = qv.LargeStringColumn()
    optFilter = qv.LargeStringColumn()
    number_obs = qv.Int64Column(nullable=True)
    min_apparent_mag = qv.Float64Column(nullable=True)
    max_apparent_mag = qv.Float64Column(nullable=True)
    median_apparent_mag = qv.Float64Column(nullable=True)
    min_phase = qv.Float64Column(nullable=True)
    max_phase = qv.Float64Column(nullable=True)


def write_sorcha_inputs(
    output_dir: str,
    small_bodies: SmallBodies,
    pointings: Pointings,
    observatory: Observatory,
    time_range: Optional[list[float]] = None,
    format: Literal["csv", "whitespace"] = "csv",
    element_type: Literal["cartesian", "keplerian", "cometary"] = "cartesian",
    orbits_file_name: str = "orbits.csv",
    properties_file_name: str = "properties.csv",
    pointings_database_name: str = "pointings.db",
    sorcha_config_file_name: str = "config.ini",
    randomization: bool = True,
    output_columns: Literal["basic", "all"] = "all",
) -> dict[str, str]:
    """
    Write the small body population, pointings, and observatory to files that can be read by Sorcha.

    Parameters
    ----------
    output_dir : str
        The directory to write the files to.
    small_bodies : SmallBodies
        The small body population to write.
    pointings : Pointings
        The pointings to write to a SQLite database.
    observatory : Observatory
        The observatory to write to a configuration file.
    time_range : list[float], optional
        The time range to filter the pointings by, by default None.
    format : Literal["csv", "whitespace"], optional
        The format to write the orbits and photometric properties in, by default "csv".
    element_type : Literal["cartesian", "keplerian", "cometary"], optional
        The type of orbital elements to write, by default "cartesian".
    orbits_file_name : str, optional
        The name of the file to write the orbits to, by default "orbits.csv".
    properties_file_name : str, optional
        The name of the file to write the photometric properties to, by default "properties.csv".
    pointings_database_name : str, optional
        The name of the SQLite database to write the pointings to, by default "pointings.db".
    sorcha_config_file_name : str, optional
        The name of the configuration file to write the observatory to, by default "config.ini".
    randomization : bool, optional
        Ramdomize the photometry and astrometry using the calculated uncertainties, by default True.
    output_columns : Literal["basic", "all"], optional
        The columns to output in the Sorcha output, by default "all".

    Returns
    -------
    paths : dict
        A dictionary containing the paths to the files with the following keys:
        - "orbits" : The path to the orbits file.
        - "photometric_properties" : The path to the photometric properties file.
        - "pointings" : The path to the pointings database.
        - "config" : The path to the configuration file.
    """
    paths = {}

    os.makedirs(output_dir, exist_ok=True)

    # Write orbits and photometric properties to the corresponding file types
    orbits_df = orbits_to_sorcha_dataframe(small_bodies.orbits, element_type)
    properties_df = photometric_properties_to_sorcha_dataframe(
        small_bodies.properties, small_bodies.orbits.object_id, observatory.main_filter
    )

    orbit_file = os.path.join(output_dir, orbits_file_name)
    properties_file = os.path.join(output_dir, properties_file_name)

    if format == "csv":
        sep = ","
    elif format == "whitespace":
        sep = " "
    else:
        raise ValueError("format must be either 'csv' or 'whitespace'")

    orbits_df.to_csv(orbit_file, index=False, sep=sep, float_format="%.15f")
    properties_df.to_csv(properties_file, index=False, sep=sep, float_format="%.15f")

    paths["orbits"] = orbit_file
    paths["photometric_properties"] = properties_file

    # Write pointings to a sqlite database
    pointings_file = os.path.join(output_dir, pointings_database_name)
    con = sql.connect(pointings_file)
    pointings.to_sql(con, table_name="pointings")
    con.close()

    paths["pointings"] = pointings_file

    # Write the observatory to a configuration file
    config_file = os.path.join(output_dir, sorcha_config_file_name)
    with open(config_file, "w") as f:
        config_str = observatory_to_sorcha_config(
            observatory,
            time_range=time_range,
            randomization=randomization,
            output_columns=output_columns,
        )
        f.write(config_str)

    paths["config"] = config_file

    return paths


def sorcha(
    output_dir: str,
    small_bodies: SmallBodies,
    pointings: Pointings,
    observatory: Observatory,
    time_range: Optional[list[float]] = None,
    tag: str = "sorcha",
    overwrite: bool = True,
    randomization: bool = True,
    output_columns: Literal["basic", "all"] = "all",
) -> tuple[Union[SorchaOutputBasic, SorchaOutputAll], SorchaOutputStats]:
    """
    Run sorcha on the given small bodies, pointings, and observatory.

    Parameters
    ----------
    output_dir : str
        The directory to write the Sorcha output to.
    small_bodies : SmallBodies
        The small body population to run Sorcha on.
    pointings : Pointings
        The pointings to run Sorcha on.
    observatory : Observatory
        The observatory to run Sorcha on.
    time_range : list[float], optional
        The time range to filter the pointings by, by default None.
    tag : str, optional
        The tag to use for the output files, by default "sorcha".
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    randomization : bool, optional
        Ramdomize the photometry and astrometry using the calculated uncertainties, by default True.
    output_columns : Literal["basic", "all"], optional
        The columns to output in the Sorcha output, by default "all".

    Returns
    -------
    tuple[Union[SorchaOutputBasic, SorchaOutputAll], SorchaOutputStats]
        Sorcha output observations (in basic or all formats) and statistics
        per object and filter.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    paths = write_sorcha_inputs(
        output_dir,
        small_bodies,
        pointings,
        observatory,
        time_range=time_range,
        randomization=randomization,
        output_columns=output_columns,
    )

    # Note that the stats file is automatically saved in the output directory and with a
    # preset file extension so we do not need to add the output directory to the stats file path
    stats_file = f"{tag}_stats"
    command = [
        "sorcha",
        "run",
        "-c",
        f"{paths['config']}",
        "-p",
        f"{paths['photometric_properties']}",
        "-ob",
        f"{paths['orbits']}",
        "-pd",
        f"{paths['pointings']}",
        "-o",
        f"{output_dir}",
        "-t",
        f"{tag}",
        "-st",
        f"{stats_file}",
    ]
    if overwrite:
        command.append("-f")

    results = subprocess.run(command, check=False, capture_output=True)
    if results.returncode != 0:
        raise RuntimeError(
            f"Sorcha failed with the following error:\n{results.stderr.decode()}"
        )
    else:
        if output_columns == "basic":
            sorcha_outputs = SorchaOutputBasic.from_csv(f"{output_dir}/{tag}.csv")
        else:
            sorcha_outputs = SorchaOutputAll.from_csv(f"{output_dir}/{tag}.csv")

        sorcha_stats = SorchaOutputStats.from_csv(f"{output_dir}/{stats_file}.csv")

    return sorcha_outputs, sorcha_stats
