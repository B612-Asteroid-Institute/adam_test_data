import glob
import os

import pandas as pd
import pyarrow as pa
import quivr as qv
from adam_core.coordinates import CometaryCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from ..populations import PhotometricProperties, SmallBodies


def read_des_from_file(file: str) -> pd.DataFrame:
    """
    Read a DES-formatted file into a pandas DataFrame.

    Accepted comment lines start with either a "!!" or a "#". The last comment line is assumed to be the header line
    which contains the column names. The data is read in using the column names and the data is returned as a
    pandas DataFrame.

    Parameters
    ----------
    file : str
        The path to the DES-formatted file.

    Returns
    -------
    des : pd.DataFrame
        The data from the DES-formatted file.
    """
    with open(file, "r") as f:
        s3m = f.readlines()

    s3m = [line.strip() for line in s3m]

    comment_lines = [
        line for line in s3m if line.startswith("!!") or line.startswith("#")
    ]
    header_line = comment_lines[-1]
    columns = header_line.strip("!!").strip("#").split()

    des = pd.read_table(
        file, sep="\s+", skiprows=len(comment_lines), names=columns, index_col=False
    )

    # The DES format has inconsistent column names so we rename them to be consistent
    des.rename(
        columns={
            "S3MID": "ObjID",
            "ObjectID": "ObjID",
            "OID": "ObjID",
            "Omega": "node",
        },
        inplace=True,
    )

    return des


def load_S3M(directory: str) -> SmallBodies:
    """
    Load the S3M small body population from a directory.

    The S3M population is stored in a series of files in the directory. Each file contains a subset of the population
    and the files are read in and concatenated together to form the full population.

    Parameters
    ----------
    directory : str
        The directory containing the S3M population files.

    Returns
    -------
    s3m : SmallBodies
        The S3M population.
    """
    s3m_files = sorted(glob.glob(os.path.join(directory, "*.s3m")))

    s3m = SmallBodies.empty(name="S3M")
    id_offset = 0
    for s3m_file in s3m_files:
        print(f"Loading {s3m_file}")

        s3m_df_i = read_des_from_file(s3m_file)

        # Create orbit IDs for the s3m population
        # There are ~14 million objects in the population so 7 digits should be enough
        orbit_ids = pa.array(
            [f"s3m{i:07d}" for i in range(id_offset, id_offset + len(s3m_df_i))]
        )
        orbits_i = Orbits.from_kwargs(
            orbit_id=orbit_ids,
            object_id=s3m_df_i["ObjID"],
            coordinates=CometaryCoordinates.from_kwargs(
                q=s3m_df_i["q"],
                e=s3m_df_i["e"],
                i=s3m_df_i["i"],
                raan=s3m_df_i["node"],
                ap=s3m_df_i["argperi"],
                tp=s3m_df_i["t_p"],
                time=Timestamp.from_mjd(s3m_df_i["t_0"].values, scale="tdb"),
                origin=Origin.from_kwargs(
                    code=pa.repeat("SUN", len(s3m_df_i)),
                ),
                frame="ecliptic",
            ).to_cartesian(),
        )

        photometric_properties_i = PhotometricProperties.from_kwargs(
            H_mf=s3m_df_i["H"],
        )

        s3m_i = SmallBodies.from_kwargs(
            orbits=orbits_i,
            properties=photometric_properties_i,
            name="S3M",
        )

        s3m = qv.concatenate([s3m, s3m_i])

        id_offset += len(s3m_df_i)

    return s3m
