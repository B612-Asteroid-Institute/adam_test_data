import pandas as pd
import pyarrow as pa
import pytest
from adam_core.coordinates import KeplerianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from ..populations import (
    PhotometricProperties,
    orbits_to_sorcha_dataframe,
    photometric_properties_to_sorcha_dataframe,
)


def test_photometric_properties_to_sorcha_dataframe():
    # Test that that the photometric properties are correctly written to a Sorcha-compatible dataframe.
    object_ids = ["1", "2", "3"]
    H_r = [20.0, 21.0, 22.0]
    u_r = [0.15, 0.20, 0.25]
    g_r = [0.70, 0.75, 0.80]
    r_r = [0.60, 0.65, 0.70]

    photometric_properties = PhotometricProperties.from_kwargs(
        H_mf=H_r,
        u_mf=u_r,
        g_mf=g_r,
        r_mf=r_r,
    )

    expected_df = pd.DataFrame(
        {
            "ObjID": object_ids,
            "H_r": H_r,
            "u-r": u_r,
            "g-r": g_r,
            "r-r": r_r,
            "i-r": [0.0, 0.0, 0.0],
            "z-r": [0.0, 0.0, 0.0],
            "y-r": [0.0, 0.0, 0.0],
            "Y-r": [0.0, 0.0, 0.0],
            "VR-r": [0.0, 0.0, 0.0],
            "GS": [0.15, 0.15, 0.15],
        }
    )

    actual_df = photometric_properties_to_sorcha_dataframe(
        photometric_properties, object_ids, main_filter="r"
    )

    pd.testing.assert_frame_equal(actual_df, expected_df)


@pytest.fixture
def orbits():
    object_ids = ["1", "2", "3"]
    keplerian_elements = KeplerianCoordinates.from_kwargs(
        a=[1.3, 2.0, 5.0],
        e=[0.1, 0.2, 0.3],
        i=[0, 30, 60],
        raan=[0, 90, 180],
        ap=[0, 90, 180],
        M=[0, 90, 180],
        time=Timestamp.from_mjd(
            pa.repeat(59580.0, 3),
            scale="tdb",
        ),
        origin=Origin.from_kwargs(code=pa.repeat("SUN", 3)),
        frame="ecliptic",
    )
    cartesian_elements = keplerian_elements.to_cartesian()

    orbits = Orbits.from_kwargs(
        orbit_id=object_ids,
        object_id=object_ids,
        coordinates=cartesian_elements,
    )
    return orbits


@pytest.mark.parametrize("element_type", ["cartesian", "keplerian", "cometary"])
def test_orbits_to_sorcha_dataframe(orbits, element_type):
    # Test that the orbits are correctly written to a Sorcha-compatible dataframe.

    if element_type == "cartesian":

        expected_df = orbits.coordinates.to_dataframe()
        expected_df.insert(0, "ObjID", orbits.object_id)
        expected_df.insert(1, "FORMAT", "CART")
        expected_df.rename(
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

        df_actual = orbits_to_sorcha_dataframe(orbits, element_type="cartesian")

    elif element_type == "keplerian":
        expected_df = orbits.coordinates.to_keplerian().to_dataframe()
        expected_df.insert(0, "ObjID", orbits.object_id)
        expected_df.insert(1, "FORMAT", "KEP")
        expected_df.rename(
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

        df_actual = orbits_to_sorcha_dataframe(orbits, element_type="keplerian")

    elif element_type == "cometary":
        expected_df = orbits.coordinates.to_cometary().to_dataframe()
        expected_df.insert(0, "ObjID", orbits.object_id)
        expected_df.insert(1, "FORMAT", "COM")
        expected_df.rename(
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

        df_actual = orbits_to_sorcha_dataframe(orbits, element_type="cometary")

    expected_df.drop(
        columns=[
            "time.days",
            "time.nanos",
            "covariance.values",
            "origin.code",
        ],
        inplace=True,
    )
    expected_df.insert(8, "epochMJD_TDB", orbits.coordinates.time.rescale("tdb").mjd())

    pd.testing.assert_frame_equal(df_actual, expected_df)
