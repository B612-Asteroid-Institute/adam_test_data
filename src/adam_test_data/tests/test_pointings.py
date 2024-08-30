import sqlite3 as sql
import tempfile

import pytest

from ..pointings import Pointings


@pytest.fixture
def pointings():

    pointings = Pointings.from_kwargs(
        observationId=[
            "c4d_120923_034041_ooi_z_a1",
            "c4d_120923_052321_ooi_r_a1",
            "c4d_120923_052740_ooi_z_a1",
            "c4d_120923_055741_ooi_z_a1",
            "c4d_120923_055938_ooi_r_a1",
        ],
        observationStartMJD_TAI=[
            56193.15366505399,
            56193.22496089093,
            56193.22795809792,
            56193.24879656812,
            56193.250160232994,
        ],
        visitTime=[30, 100, 100, 30, 30],
        visitExposureTime=[30, 30, 30, 30, 30],
        filter=["z", "r", "z", "z", "r"],
        seeingFwhmGeom_arcsec=[
            1.730654,
            1.869076,
            1.877529,
            1.827344,
            1.816441,
        ],
        seeingFwhmEff_arcsec=[
            1.730654,
            1.869076,
            1.877529,
            1.827344,
            1.816441,
        ],
        fieldFiveSigmaDepth_mag=[
            21.139450625,
            22.59583225,
            21.850898625,
            21.285844875,
            22.178669125,
        ],
        fieldRA_deg=[
            325.45847092705367,
            0.0613164241568791,
            0.0616686860351492,
            13.722163671721958,
            13.72215330465642,
        ],
        fieldDec_deg=[
            0.5735200759194536,
            -0.0869715522298139,
            -0.0868865703293122,
            0.5847894812624722,
            0.5849129283284105,
        ],
        rotSkyPos_deg=[0.0, 0.0, 0.0, 0.0, 0.0],
        observatory_code=[
            "W84",
            "W84",
            "W84",
            "W84",
            "W84",
        ],
    )

    return pointings


@pytest.fixture
def pointings_db():
    temp_db = tempfile.NamedTemporaryFile(suffix=".db")
    temp_db.close()
    yield temp_db.name


def test_pointings_to_from_sql(pointings, pointings_db):
    # Test that we can save and load the pointings table to and from an SQLite database.
    con = sql.connect(pointings_db)
    pointings.to_sql(con, table_name="test_pointings")

    pointings_from_sql = Pointings.from_sql(con, table_name="test_pointings")

    assert pointings == pointings_from_sql
