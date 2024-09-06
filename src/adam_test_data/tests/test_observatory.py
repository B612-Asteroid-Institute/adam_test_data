import pytest

from ..observatory import (
    FieldOfView,
    Observatory,
    Simulation,
    observatory_to_sorcha_config,
)


def test_FieldOfView_to_string():
    # Test that FieldOfView.to_string returns the correct string representation of the object.
    # These strings are used to create sorcha configuration files.
    fov = FieldOfView(camera_model="circle", fill_factor=0.5, circle_radius=1.0)

    assert (
        fov.to_string()
        == "[FOV]\ncamera_model = circle\nfill_factor = 0.5\ncircle_radius = 1.0"
    )

    fov = FieldOfView(camera_model="footprint", footprint_path="path/to/footprint")

    assert (
        fov.to_string()
        == "[FOV]\ncamera_model = footprint\nfootprint_path = path/to/footprint"
    )

    fov = FieldOfView(
        camera_model="footprint",
        footprint_path="path/to/footprint",
        footprint_edge_threshold=0.1,
    )

    assert (
        fov.to_string()
        == "[FOV]\ncamera_model = footprint\nfootprint_path = path/to/footprint\nfootprint_edge_threshold = 0.1"
    )


def test_FieldOfView_raises():
    # Test that FieldOfView raises the correct exceptions when invalid argument combinations
    # are passed.
    with pytest.raises(ValueError, match="Unknown camera model: unknown"):
        FieldOfView(camera_model="unknown")

    with pytest.raises(
        ValueError, match="fill_factor is required for camera_model='circle'"
    ):
        FieldOfView(camera_model="circle", circle_radius=1.0)

    with pytest.raises(
        ValueError, match="circle_radius is required for camera_model='circle'"
    ):
        FieldOfView(camera_model="circle", fill_factor=0.5)

    with pytest.raises(
        ValueError, match="footprint_path is required for camera_model='footprint'"
    ):
        FieldOfView(camera_model="footprint")

    with pytest.raises(
        ValueError,
        match="footprint_edge_threshold is only valid for camera_model='footprint'",
    ):
        FieldOfView(
            camera_model="circle",
            fill_factor=0.5,
            circle_radius=1.0,
            footprint_edge_threshold=0.1,
        )


def test_Simulation_to_string():
    # Test that Simulation.to_string returns the correct string representation of the object.
    # These strings are used to create sorcha configuration files.
    sim = Simulation(ang_fov=1.0, fov_buffer=0.1)

    assert (
        sim.to_string("500")
        == """[SIMULATION]
ar_ang_fov = 1.0
ar_fov_buffer = 0.1
ar_picket = 1
ar_obs_code = 500
ar_healpix_order = 6"""
    )

    sim = Simulation(ang_fov=1.0, fov_buffer=0.1, picket=2, healpix_order=7)

    assert (
        sim.to_string("X05")
        == """[SIMULATION]
ar_ang_fov = 1.0
ar_fov_buffer = 0.1
ar_picket = 2
ar_obs_code = X05
ar_healpix_order = 7"""
    )


def test_observatory_to_sorcha_config():

    # Test that observatory_to_sorcha_config returns the correct string representation of the
    # object. These strings are used to create sorcha configuration files.
    obs = Observatory(
        code="X05",
        filters=["u", "g", "r", "i", "z", "y"],
        bright_limit=[16, 16, 16, 16, 16, 16],
        fov=FieldOfView(camera_model="circle", fill_factor=0.9, circle_radius=3),
        simulation=Simulation(ang_fov=1.0, fov_buffer=0.1),
        main_filter="r",
    )

    assert (
        observatory_to_sorcha_config(obs)
        == """
# Sorcha Configuration File - ADAM Test Data - X05

[INPUT]
ephemerides_type = ar
eph_format = csv
size_serial_chunk = 5000
aux_format = csv
pointing_sql_query = SELECT * FROM pointings WHERE observatory_code = 'X05'

[SIMULATION]
ar_ang_fov = 1.0
ar_fov_buffer = 0.1
ar_picket = 1
ar_obs_code = X05
ar_healpix_order = 6

[FILTERS]
observing_filters = r,u,g,i,z,y

[SATURATION]
bright_limit = 16,16,16,16,16,16

[PHASECURVES]
phase_function = HG

[FOV]
camera_model = circle
fill_factor = 0.9
circle_radius = 3

[FADINGFUNCTION]
fading_function_on = True
fading_function_width = 0.1
fading_function_peak_efficiency = 1

[OUTPUT]
output_format = csv
output_columns = basic

[LIGHTCURVE]
lc_model = none

[ACTIVITY]
comet_activity = none

[EXPERT]
randomization_on = True
vignetting_on = True
trailing_losses_on = True
"""
    )


def test_observatory_to_sorcha_config_raises():
    # Test that observatory_to_sorcha_config raises the correct exceptions when invalid argument
    # combinations are passed.

    with pytest.raises(ValueError, match="Main filter r not in list of filters"):
        obs = Observatory(
            code="X05",
            filters=["u", "g", "i", "z", "y"],
            bright_limit=[16, 16, 16, 16, 16, 16],
            fov=FieldOfView(camera_model="circle", fill_factor=0.9, circle_radius=3),
            simulation=Simulation(ang_fov=1.0, fov_buffer=0.1),
            main_filter="r",
        )
        observatory_to_sorcha_config(obs)
