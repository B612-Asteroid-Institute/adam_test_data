import os
import tempfile

import pyarrow as pa
import pytest
from adam_core.coordinates import CartesianCoordinates
from adam_core.coordinates.origin import Origin
from adam_core.orbits import Orbits
from adam_core.time import Timestamp

from ..main import write_sorcha_inputs
from ..observatory import FieldOfView, Observatory, Simulation
from ..pointings import Pointings
from ..populations import PhotometricProperties, SmallBodies


@pytest.fixture
def small_bodies():
    # Simple MBA pulled from SBDB using adam_core:
    # from adam_core.orbits.query import query_sbdb
    # orbits = query_sbdb(["2013 RR165"])
    orbits = Orbits.from_kwargs(
        orbit_id=["t00001"],
        object_id=["(2013 RR165)"],
        coordinates=CartesianCoordinates.from_kwargs(
            x=[-0.41038696631631916],
            y=[2.6829549339299468],
            z=[0.05748577701259661],
            vx=[-0.01012180471216174],
            vy=[-0.0008402902010790527],
            vz=[-0.00042127722432239693],
            time=Timestamp.from_kwargs(days=[56981], nanos=[0], scale="tdb"),
            origin=Origin.from_kwargs(code=["SUN"]),
            frame="ecliptic",
        ),
    )

    properties = PhotometricProperties.from_kwargs(
        H_mf=[18.61],
    )

    return SmallBodies.from_kwargs(orbits=orbits, properties=properties)


@pytest.fixture
def pointings():
    # Generated using adam_core:
    # import numpy as np
    # from astropy import units as u
    #
    # from adam_core.orbits.query import query_sbdb
    # from adam_core.propagator.adam_pyoorb import PYOORBPropagator
    # from adam_core.observers import Observers
    # from adam_core.time import Timestamp
    #
    # # Get orbits to propagate
    # initial_time = Timestamp.from_mjd([70000.0], scale="tdb")
    # object_ids = ["2013 RR165"]
    # orbits = query_sbdb(object_ids)
    #
    # # Make sure PYOORB is ready
    # propagator = PYOORBPropagator()
    #
    # # Define a set of observers and observation times
    # times = Timestamp.from_mjd(initial_time.mjd() + np.arange(0, 12, 2))
    # observers = Observers.from_code("X05", times)
    #
    # # Generate ephemerides! This function supports multiprocessing for large
    # # propagation jobs.
    # ephemeris = propagator.generate_ephemeris(
    #     orbits,
    #     observers,
    #     chunk_size=100,
    #     max_processes=1
    # )
    observation_times = Timestamp.from_kwargs(
        days=[70000, 70002, 70004, 70006, 70008, 70010],
        nanos=[0, 0, 0, 0, 0, 0],
        scale="tai",
    )
    num_visits = len(observation_times)

    return Pointings.from_kwargs(
        observationId=[f"exp{i:02d}" for i in range(num_visits)],
        observationStartMJD_TAI=observation_times.mjd(),
        visitTime=pa.repeat(34, num_visits),
        visitExposureTime=pa.repeat(30, num_visits),
        filter=["u", "g", "r", "i", "z", "y"],
        seeingFwhmGeom_arcsec=pa.repeat(0.25, num_visits),
        seeingFwhmEff_arcsec=pa.repeat(0.25, num_visits),
        fieldFiveSigmaDepth_mag=pa.repeat(24.0, num_visits),
        # These are the predicted positions of 2013 RR165 at the observation times
        # as observed from the Rubin Observatory LSST
        fieldRA_deg=[
            313.23563951677465,
            312.8526829837726,
            312.4507127362583,
            312.0317168141278,
            311.5978588440544,
            311.15144471605623,
        ],
        fieldDec_deg=[
            -17.699419833054964,
            -17.75604384734044,
            -17.816253170916713,
            -17.879473511313485,
            -17.945108182773343,
            -18.01255021852454,
        ],
        rotSkyPos_deg=pa.repeat(0.0, num_visits),
    )


@pytest.fixture
def observatory():
    # Rubin Observatory's LSST modeled with a circular footprint
    # From: https://sorcha.readthedocs.io/en/latest/configfiles.html#rubin-circular-approximation
    return Observatory(
        code="X05",
        filters=["u", "g", "r", "i", "z", "y"],
        main_filter="r",
        bright_limit=[16.0, 16.0, 16.0, 16.0, 16.0, 16.0],
        fov=FieldOfView(
            camera_model="circle",
            circle_radius=1.75,
            fill_factor=0.9,
        ),
        simulation=Simulation(ang_fov=2.06, fov_buffer=0.2),
    )


def test_write_sorcha_inputs(small_bodies, pointings, observatory):
    # Test that write_sorcha_inputs writes the expected files to the expected locations
    # The actual contents of the files are not checked here but in different tests
    with tempfile.TemporaryDirectory() as temp_dir:

        paths = write_sorcha_inputs(
            small_bodies=small_bodies,
            pointings=pointings,
            observatory=observatory,
            output_dir=temp_dir,
        )

        # Check that the expected files were written
        assert paths["orbits"] == f"{temp_dir}/orbits.csv"
        assert paths["photometric_properties"] == f"{temp_dir}/properties.csv"
        assert paths["pointings"] == f"{temp_dir}/pointings.db"
        assert paths["config"] == f"{temp_dir}/config.ini"
        for k, v in paths.items():
            assert os.path.exists(v)

        paths = write_sorcha_inputs(
            small_bodies=small_bodies,
            pointings=pointings,
            observatory=observatory,
            output_dir=temp_dir,
            format="whitespace",
            orbits_file_name="orbits.txt",
            properties_file_name="properties.txt",
            pointings_database_name="pointings.sqlite",
            sorcha_config_file_name="configuration.ini",
        )

        # Check that the expected files were written
        assert paths["orbits"] == f"{temp_dir}/orbits.txt"
        assert paths["photometric_properties"] == f"{temp_dir}/properties.txt"
        assert paths["pointings"] == f"{temp_dir}/pointings.sqlite"
        assert paths["config"] == f"{temp_dir}/configuration.ini"
        for k, v in paths.items():
            assert os.path.exists(v)
