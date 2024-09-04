import os
import sqlite3 as sql
from typing import Literal, Optional

from .observatory import Observatory, observatory_to_sorcha_config
from .pointings import Pointings
from .populations import (
    SmallBodies,
    orbits_to_sorcha_dataframe,
    photometric_properties_to_sorcha_dataframe,
)


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
        config_str = observatory_to_sorcha_config(observatory, time_range=time_range)
        f.write(config_str)

    paths["config"] = config_file

    return paths
