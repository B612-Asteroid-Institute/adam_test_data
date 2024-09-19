import os

import numpy as np
import pandas as pd
import pyarrow as pa
from adam_core.coordinates import KeplerianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from ..populations import PhotometricProperties, SmallBodies


def load_P9(directory: str) -> SmallBodies:
    """
    Load the planet 9 population model from Brown and Batygin (2021a).

    The model can be downloaded from: https://data.caltech.edu/records/8fjad-x7y61

    Parameters
    ----------
    directory : str
        The directory containing the planet 9 population model.

    Returns
    -------
    p9 : SmallBodies
        The planet 9 population model.
    """
    p9 = pd.read_csv(os.path.join(directory, "reference_population.csv"), comment="#")
    p9.columns = p9.columns.str.strip()

    # From the comments in the file
    time = Timestamp.from_jd([2458270.0 for _ in range(len(p9))], scale="tdb")
    orbit_ids = pa.array([f"P9_{i:07d}" for i in range(len(p9))])
    orbits = Orbits.from_kwargs(
        orbit_id=orbit_ids,
        object_id=np.array(p9.index.values.astype(str), dtype="object"),
        coordinates=KeplerianCoordinates.from_kwargs(
            a=p9["a"],
            e=p9["e"],
            i=p9["inc"],
            raan=p9["Omega"],
            ap=p9["varpi"],
            M=p9["M"],
            time=time,
            origin=Origin.from_kwargs(code=pa.repeat("SUN", len(p9))),
            frame="equatorial",
        ).to_cartesian(),
    )

    photometric_properties = PhotometricProperties.from_kwargs(
        H_mf=p9["V"],
    )

    return SmallBodies.from_kwargs(
        orbits=orbits,
        properties=photometric_properties,
        name="P9",
    )
