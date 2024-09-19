from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc
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
    name = qv.StringAttribute()


def photometric_properties_to_sorcha_table(
    properties: PhotometricProperties, object_ids: pa.Array, main_filter: str
) -> pa.Table:
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
    table : pa.Table
        The photometric properties in a pyarrow Table that can be serialized
        to a CSV or a white-space separated text file.
    """
    table = properties.table
    table = table.add_column(0, "ObjID", object_ids)

    # Map the columns to the correct filter. So all _mf colors
    # will be renamed to {filter}-{main_filter}. The absolute magnitude
    # will be renamed to H_{main_filter} (note an underscore instead of a dash)
    column_names = table.column_names
    new_names = []
    for c in column_names:
        if c.endswith("_mf"):
            if c == "H_mf":
                new_name = f"H_{main_filter}"
            else:
                new_name = c.replace("_mf", f"-{main_filter}")
        else:
            new_name = c

        new_names.append(new_name)

    table = table.rename_columns(new_names)
    return table


def orbits_to_sorcha_table(
    orbits: Orbits, element_type: Literal["cartesian", "keplerian", "cometary"]
) -> pa.Table:
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
    table : pa.Table
        The orbits in a pyarrow Table that can be serialized
        to a CSV or a white-space separated text file.
    """
    if element_type == "cartesian":
        table = orbits.coordinates.table
        table = table.drop_columns(["time", "covariance", "origin"])
        table = table.rename_columns(["x", "y", "z", "xdot", "ydot", "zdot"])
        format = "CART"

    elif element_type == "keplerian":
        table = orbits.coordinates.to_keplerian().table
        table = table.drop_columns(["time", "covariance", "origin"])
        table = table.rename_columns(["a", "e", "inc", "node", "argPeri", "ma"])
        format = "KEP"

    elif element_type == "cometary":
        table = orbits.coordinates.to_cometary().table
        table = table.drop_columns(["time", "covariance", "origin"])
        table = table.rename_columns(
            ["q", "e", "inc", "node", "argPeri", "t_p_MJD_TDB"]
        )
        format = "COM"
    else:
        raise ValueError(f"Unknown element type: {element_type}")

    table = table.add_column(0, "ObjID", orbits.object_id)
    table = table.add_column(
        1, "FORMAT", pc.cast(pa.repeat(format, len(orbits)), pa.large_string())
    )
    table = table.add_column(
        8, "epochMJD_TDB", orbits.coordinates.time.rescale("tdb").mjd()
    )

    return table
