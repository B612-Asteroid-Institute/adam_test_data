import os
import tempfile

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from ..p9 import load_P9


@pytest.fixture
def P9() -> str:
    # head -n 15 reference_population.csv
    return """# Planet Nine reference population; epoch of JD=2458270.0 (1 June 2018)
# Orbital elements refer to J2000 heliocentric osculating elements
#Parameters are:
# index, mass (Earth masses), semimajor axis (AU), eccentricity, inclination (deg), longitude of perihelion (deg), longitude of ascending node (deg), mean anomaly at epoch (deg), R.A. at epoch (deg), declination at epoch (deg), heliocentric distance at epoch (AU), assumed radius (Earth radii), assumed albedo, V magnitude at epoch, detectable in ZTF survey? (0=no/1=yes), detectable in DES survey (0=no/1=yes), detectable in PS1 (0=no/1=yes)
index, mass, a, e, inc, varpi, Omega, M, ra, dec, R, rad, albedo, V, ZTF, DES, PS1
    0,  4.84,  249.58,  0.074,  12.50,   252.85,    48.03,   72.60,  340.9044,  -21.0693,  245.34,  1.61,  0.270,  19.39, 1, 0, 1
    1,  7.54,  375.45,  0.232,  14.13,   237.85,   111.55,   55.36,  322.4211,  -21.4985,  342.36,  2.51,  0.483,  19.24, 1, 0, 1
    2, 10.62,  944.92,  0.502,  18.60,   237.12,    95.17,   71.09,   12.2508,  -14.9859, 1022.87,  3.54,  0.732,  22.80, 0, 0, 0
    3,  5.34,  517.69,  0.605,  20.88,   229.64,    94.30,  204.74,   59.7006,    7.9102,  819.32,  1.78,  0.694,  23.39, 0, 0, 0
    4,  5.13,  337.08,  0.083,  17.94,   251.60,    68.47,  134.16,   34.8625,    2.8018,  357.57,  1.71,  0.343,  20.64, 1, 1, 1
    5,  3.78,  278.45,  0.317,  20.10,   247.90,   121.33,   99.15,   24.4436,  -10.8979,  316.44,  1.26,  0.588,  20.19, 1, 1, 1
    6,  4.74,  275.19,  0.151,  18.73,   259.40,    98.51,  264.27,  152.7116,   26.1338,  285.30,  1.58,  0.693,  19.07, 1, 0, 1
    7,  6.36,  504.83,  0.356,  21.51,   233.43,   105.46,   69.63,  351.4116,  -23.6505,  504.85,  2.12,  0.628,  21.02, 0, 0, 0
    8,  9.64,  391.04,  0.226,  11.97,   246.98,    76.04,  128.71,   33.9063,    4.9301,  456.29,  3.21,  0.407,  20.15, 1, 1, 1
    9,  5.42,  364.43,  0.127,  17.13,   234.72,   123.50,  292.78,  157.2145,   18.5867,  351.84,  1.81,  0.712,  19.66, 1, 0, 1
"""


def test_load_P9(P9: str) -> None:

    with tempfile.TemporaryDirectory() as p9_dir:

        with open(os.path.join(p9_dir, "reference_population.csv"), "w") as f:
            f.write(P9)

        small_bodies = load_P9(p9_dir)
        assert len(small_bodies) == 10
        assert pc.equal(
            small_bodies.orbits.orbit_id,
            pa.array(
                [
                    "P9_0000000",
                    "P9_0000001",
                    "P9_0000002",
                    "P9_0000003",
                    "P9_0000004",
                    "P9_0000005",
                    "P9_0000006",
                    "P9_0000007",
                    "P9_0000008",
                    "P9_0000009",
                ]
            ),
        )

        assert pc.equal(
            small_bodies.orbits.object_id,
            pa.array(
                [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                ]
            ),
        )
