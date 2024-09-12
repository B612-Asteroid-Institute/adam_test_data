from typing import Literal

import pyarrow as pa
import pytest
from adam_core.coordinates import KeplerianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from ..populations import (
    PhotometricProperties,
    orbits_to_sorcha_table,
    photometric_properties_to_sorcha_table,
)


def test_photometric_properties_to_sorcha_table() -> None:
    # Test that that the photometric properties are correctly written to a Sorcha-compatible dataframe.
    object_ids = pa.array(["1", "2", "3"])
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

    expected_table = pa.Table.from_pydict(
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
        },
        schema=pa.schema(
            [
                pa.field("ObjID", pa.string()),
                pa.field("H_r", pa.float64(), nullable=False),
                pa.field("u-r", pa.float64(), nullable=False),
                pa.field("g-r", pa.float64(), nullable=False),
                pa.field("r-r", pa.float64(), nullable=False),
                pa.field("i-r", pa.float64(), nullable=False),
                pa.field("z-r", pa.float64(), nullable=False),
                pa.field("y-r", pa.float64(), nullable=False),
                pa.field("Y-r", pa.float64(), nullable=False),
                pa.field("VR-r", pa.float64(), nullable=False),
                pa.field("GS", pa.float64(), nullable=False),
            ]
        ),
    )

    actual_table = photometric_properties_to_sorcha_table(
        photometric_properties, object_ids, main_filter="r"
    )

    assert actual_table.equals(expected_table)


@pytest.fixture
def orbits() -> Orbits:
    object_ids = ["1", "2", "3"]
    keplerian_elements = KeplerianCoordinates.from_kwargs(
        a=[1.3, 2.0, 5.0],
        e=[0, 0.2, 0.3],
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
def test_orbits_to_sorcha_table(
    orbits: Orbits, element_type: Literal["cartesian", "keplerian", "cometary"]
) -> None:
    # Test that the orbits are correctly written to a Sorcha-compatible dataframe.

    if element_type == "cartesian":

        coordinates = orbits.coordinates
        expected_table = pa.Table.from_pydict(
            {
                "ObjID": pa.array(["1", "2", "3"]),
                "FORMAT": pa.array(["CART", "CART", "CART"]),
                "x": coordinates.x,
                "y": coordinates.y,
                "z": coordinates.z,
                "xdot": coordinates.vx,
                "ydot": coordinates.vy,
                "zdot": coordinates.vz,
                "epochMJD_TDB": pa.array([59580.0, 59580.0, 59580.0]),
            },
            schema=pa.schema(
                [
                    pa.field("ObjID", pa.large_string()),
                    pa.field("FORMAT", pa.large_string()),
                    pa.field("x", pa.float64()),
                    pa.field("y", pa.float64()),
                    pa.field("z", pa.float64()),
                    pa.field("xdot", pa.float64()),
                    pa.field("ydot", pa.float64()),
                    pa.field("zdot", pa.float64()),
                    pa.field("epochMJD_TDB", pa.float64()),
                ]
            ),
        )

        actual_table = orbits_to_sorcha_table(orbits, element_type="cartesian")

    elif element_type == "keplerian":

        coordinates = orbits.coordinates.to_keplerian()
        expected_table = pa.Table.from_pydict(
            {
                "ObjID": pa.array(["1", "2", "3"]),
                "FORMAT": pa.array(["KEP", "KEP", "KEP"]),
                "a": coordinates.a,
                "e": coordinates.e,
                "inc": coordinates.i,
                "node": coordinates.raan,
                "argPeri": coordinates.ap,
                "ma": coordinates.M,
                "epochMJD_TDB": pa.array([59580.0, 59580.0, 59580.0]),
            },
            schema=pa.schema(
                [
                    pa.field("ObjID", pa.large_string()),
                    pa.field("FORMAT", pa.large_string()),
                    pa.field("a", pa.float64(), nullable=False),
                    pa.field("e", pa.float64(), nullable=False),
                    pa.field("inc", pa.float64(), nullable=False),
                    pa.field("node", pa.float64(), nullable=False),
                    pa.field("argPeri", pa.float64(), nullable=False),
                    pa.field("ma", pa.float64(), nullable=False),
                    pa.field("epochMJD_TDB", pa.float64()),
                ]
            ),
        )

        actual_table = orbits_to_sorcha_table(orbits, element_type="keplerian")

    elif element_type == "cometary":
        coordinates = orbits.coordinates.to_cometary()
        expected_table = pa.Table.from_pydict(
            {
                "ObjID": pa.array(["1", "2", "3"]),
                "FORMAT": pa.array(["COM", "COM", "COM"]),
                "q": coordinates.q,
                "e": coordinates.e,
                "inc": coordinates.i,
                "node": coordinates.raan,
                "argPeri": coordinates.ap,
                "t_p_MJD_TDB": coordinates.tp,
                "epochMJD_TDB": pa.array([59580.0, 59580.0, 59580.0]),
            },
            schema=pa.schema(
                [
                    pa.field("ObjID", pa.large_string()),
                    pa.field("FORMAT", pa.large_string()),
                    pa.field("q", pa.float64(), nullable=False),
                    pa.field("e", pa.float64(), nullable=False),
                    pa.field("inc", pa.float64(), nullable=False),
                    pa.field("node", pa.float64(), nullable=False),
                    pa.field("argPeri", pa.float64(), nullable=False),
                    pa.field("t_p_MJD_TDB", pa.float64(), nullable=False),
                    pa.field("epochMJD_TDB", pa.float64()),
                ]
            ),
        )

        actual_table = orbits_to_sorcha_table(orbits, element_type="cometary")

    assert actual_table.equals(expected_table, check_metadata=True)
