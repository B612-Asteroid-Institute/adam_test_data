from typing import Literal

import pandas as pd
import pyarrow as pa
import quivr as qv
from adam_core.orbits import Orbits


class PhotometricProperties(qv.Table):
    H_mf = qv.Float64Column(default=0.0)
    u_mf = qv.Float64Column(default=0.0)
    g_mf = qv.Float64Column(default=0.0)
    r_mf = qv.Float64Column(default=0.0)
    i_mf = qv.Float64Column(default=0.0)
    z_mf = qv.Float64Column(default=0.0)
    y_mf = qv.Float64Column(default=0.0)
    Y_mf = qv.Float64Column(default=0.0)
    VR_mf = qv.Float64Column(default=0.0)
    GS = qv.Float64Column(default=0.15)


class SmallBodies(qv.Table):
    orbits = Orbits.as_column()
    properties = PhotometricProperties.as_column()


def photometric_properties_to_sorcha_dataframe(
    properties: PhotometricProperties, object_ids: pa.Array, main_filter: str
) -> pd.DataFrame:
    """
    Write photometric properties to a Sorcha-compatible dataframe. This dataframe
    can be either serialized to a CSV or a white-space separated text file for sorcha to
    use as input.

    The corresponding sorcha configuration file parameter is aux_format = csv or aux_format = whitespace.

    Parameters
    ----------
    properties : PhotometricProperties
        The photometric properties to write.
    object_ids : pa.Array
        The object IDs for the photometric properties. These should be in the
        same order as the photometric properties.
    main_filter : str
        The main filter for the photometric properties. The colors are expressed as relative
        to this filter. As an example, u_mf would be u-r if the main filter is r.

    Returns
    -------
    df : pd.DataFrame
        The orbits in a pandas DataFrame that can be serialized
        to a CSV or a white-space separated text file.
    """
    df = properties.to_dataframe()
    df.insert(0, "ObjID", object_ids)

    # Map the columns to the correct filter. So all _mf colors
    # will be renamed to {filter}-{main_filter}. The absolute magnitude
    # will be renamed to H_{main_filter} (note an underscore instead of a dash)
    columns_map = {}
    for c in df.columns:
        if c.endswith("_mf"):
            if c == "H_mf":
                columns_map[c] = f"H_{main_filter}"
            else:
                columns_map[c] = c.replace("_mf", f"-{main_filter}")

    df.rename(columns=columns_map, inplace=True)
    return df


def orbits_to_sorcha_dataframe(
    orbits: Orbits, element_type: Literal["cartesian", "keplerian", "cometary"]
) -> pd.DataFrame:
    """
    Write orbits to a Sorcha-compatible dataframe. This dataframe
    can be either serialized to a CSV or a white-space separated text file for sorcha to
    use as input.

    The corresponding sorcha configuration file parameter is aux_format = csv or aux_format = whitespace.

    Parameters
    ----------
    orbits : Orbits
        The orbits to write to the dataframe.
    element_type : Literal["cartesian", "keplerian", "cometary"]
        The type of elements to write to the dataframe.

    Returns
    -------
    df : pd.DataFrame
        The orbits in a pandas DataFrame that can be serialized
        to a CSV or a white-space separated text file.
    """
    if element_type == "cartesian":
        df = orbits.coordinates.to_dataframe()
        df.rename(
            columns={
                "x": "x",
                "y": "y",
                "z": "z",
                "vx": "xdot",
                "vy": "ydot",
                "vz": "zdot",
            },
            inplace=True,
        )
        format = "CART"

    elif element_type == "keplerian":
        df = orbits.coordinates.to_keplerian().to_dataframe()
        df.rename(
            columns={
                "a": "a",
                "e": "e",
                "i": "inc",
                "raan": "node",
                "ap": "argPeri",
                "m": "ma",
            },
            inplace=True,
        )
        format = "KEP"

    elif element_type == "cometary":
        df = orbits.coordinates.to_cometary().to_dataframe()
        df.rename(
            columns={
                "q": "q",
                "e": "e",
                "i": "inc",
                "raan": "node",
                "ap": "argPeri",
                "tp": "t_p_MJD_TDB",
            },
            inplace=True,
        )
        format = "COM"

    else:
        raise ValueError(f"Unknown element type: {element_type}")

    df.insert(0, "ObjID", orbits.object_id)
    df.insert(1, "FORMAT", format)
    df.drop(
        columns=["time.days", "time.nanos", "covariance.values", "origin.code"],
        inplace=True,
    )
    df.insert(8, "epochMJD_TDB", orbits.coordinates.time.rescale("tdb").mjd())

    return df
