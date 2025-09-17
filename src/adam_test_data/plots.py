from typing import Optional

import numpy as np
import plotly.graph_objects as go
from adam_core.coordinates import CartesianCoordinates, transform_coordinates
from adam_core.coordinates.origin import OriginCodes
from adam_core.observers import Observers
from adam_core.time import Timestamp

from .survey import Survey, SurveyFootprint, SurveyPointings

__all__ = ["plot_survey_footprint", "plot_survey_pointings", "plot_night_sky_evolution"]


def _create_individual_exposure_groups(exposure_ids, ra, dec, filters, times_mjd):
    """Create one group per exposure."""
    groups = {}

    # Ensure times_mjd is numpy array
    if not isinstance(times_mjd, np.ndarray):
        times_mjd = np.array(times_mjd)

    # Sort by time
    time_order = np.argsort(times_mjd)

    for i, idx in enumerate(time_order):
        # Convert MJD to readable time - now works with numpy
        time_str = f"MJD {times_mjd[idx]:.4f}"
        step_name = f"Exp {i+1:03d}"

        groups[step_name] = {
            "ra": [ra[idx]],
            "dec": [dec[idx]],
            "exposure_ids": [exposure_ids[idx]],
            "filters": [filters[idx]],
            "times_str": [time_str],
            "labels": [f"{i+1}"],
        }

    return groups


def _create_time_bundled_groups(
    exposure_ids, ra, dec, filters, times_mjd, bundle_minutes
):
    """Bundle exposures within time windows."""
    groups = {}

    # Ensure times_mjd is numpy array
    if not isinstance(times_mjd, np.ndarray):
        times_mjd = np.array(times_mjd)

    # Sort by time
    time_order = np.argsort(times_mjd)
    sorted_times = times_mjd[time_order]

    # Create time bins
    bundle_days = bundle_minutes / (24 * 60)  # Convert to days

    current_group = []
    group_start_time = sorted_times[0]
    group_counter = 1

    for i, time_idx in enumerate(time_order):
        current_time = times_mjd[time_idx]

        # Check if we should start a new group
        if current_time - group_start_time > bundle_days and current_group:
            # Save current group
            _save_time_group(
                groups,
                current_group,
                group_counter,
                exposure_ids,
                ra,
                dec,
                filters,
                times_mjd,
            )

            # Start new group
            current_group = [time_idx]
            group_start_time = current_time
            group_counter += 1
        else:
            current_group.append(time_idx)

    # Save final group
    if current_group:
        _save_time_group(
            groups,
            current_group,
            group_counter,
            exposure_ids,
            ra,
            dec,
            filters,
            times_mjd,
        )

    return groups


def _save_time_group(
    groups, indices, group_num, exposure_ids, ra, dec, filters, times_mjd
):
    """Save a time group."""
    group_name = f"Group {group_num:02d}"

    # Ensure times_mjd is numpy array for indexing
    if not isinstance(times_mjd, np.ndarray):
        times_mjd = np.array(times_mjd)

    groups[group_name] = {
        "ra": [ra[i] for i in indices],
        "dec": [dec[i] for i in indices],
        "exposure_ids": [exposure_ids[i] for i in indices],
        "filters": [filters[i] for i in indices],
        "times_str": [f"MJD {times_mjd[i]:.4f}" for i in indices],
        "labels": [str(j + 1) for j in range(len(indices))],
    }


def _get_filter_colors(filters):
    """Get colors for different filters."""
    filter_colors = {
        "u": "purple",
        "g": "blue",
        "r": "red",
        "i": "orange",
        "z": "darkred",
        "y": "black",
        "Y": "black",
    }

    return [filter_colors.get(f.lower(), "gray") for f in filters]


def _create_fov_circle(ra_center, dec_center, radius_deg, n_points=30):
    """Create FOV circle points using spherical trigonometry."""
    try:
        # Convert to radians
        ra_rad = np.radians(ra_center)
        dec_rad = np.radians(dec_center)
        radius_rad = np.radians(radius_deg)

        # Create circle
        theta = np.linspace(0, 2 * np.pi, n_points)

        # Spherical trigonometry
        cos_r = np.cos(radius_rad)
        sin_r = np.sin(radius_rad)
        cos_dec = np.cos(dec_rad)
        sin_dec = np.sin(dec_rad)

        # Calculate circle points
        dec_circle = np.arcsin(sin_dec * cos_r + cos_dec * sin_r * np.cos(theta))

        y = sin_r * np.sin(theta)
        x = cos_dec * cos_r - sin_dec * sin_r * np.cos(theta)

        ra_circle = ra_rad + np.arctan2(y, x)

        # Convert to degrees and ensure proper range
        ra_circle_deg = np.degrees(ra_circle)
        dec_circle_deg = np.degrees(dec_circle)

        ra_circle_deg = np.where(
            ra_circle_deg > 180, ra_circle_deg - 360, ra_circle_deg
        )
        ra_circle_deg = np.where(
            ra_circle_deg < -180, ra_circle_deg + 360, ra_circle_deg
        )

        # Close circle
        ra_circle_deg = np.append(ra_circle_deg, ra_circle_deg[0])
        dec_circle_deg = np.append(dec_circle_deg, dec_circle_deg[0])

        return np.column_stack([ra_circle_deg, dec_circle_deg])

    except (ValueError, RuntimeWarning):
        return None


def plot_survey_footprint(
    survey_footprint: SurveyFootprint,
    fov_radius_deg: float = 1.5,
    max_circles: Optional[int] = None,
    **kwargs,
) -> go.Figure:
    """
    Plot survey footprint with FOV circles around each pixel center.

    Parameters
    ----------
    survey_footprint : SurveyFootprint
        The survey footprint to plot.
    fov_radius_deg : float
        The FOV radius (in degrees).
    max_circles : Optional[int]
        The maximum number of circles to plot.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The figure with the survey footprint and FOV circles.
    """
    # Create synthetic pointings from footprint pixels
    ra_pixels = survey_footprint.ra.to_numpy()
    dec_pixels = survey_footprint.dec.to_numpy()

    # Limit number of circles for performance
    n_pixels = len(ra_pixels)
    if max_circles is None:
        max_circles = n_pixels

    if n_pixels > max_circles:
        # Sample evenly across pixels
        indices = np.linspace(0, n_pixels - 1, max_circles, dtype=int)
        ra_selected = ra_pixels[indices]
        dec_selected = dec_pixels[indices]
    else:
        ra_selected = ra_pixels
        dec_selected = dec_pixels

    # Create figure
    fig = go.Figure()

    # Plot footprint pixels
    ra_plot = np.where(ra_pixels > 180, ra_pixels - 360, ra_pixels)
    fig.add_trace(
        go.Scattergeo(
            lon=ra_plot,
            lat=dec_pixels,
            mode="markers",
            marker=dict(size=3, color="lightblue", opacity=0.6, symbol="circle"),
            name="Survey Footprint",
            showlegend=True,
            hovertemplate="RA: %{lon:.2f}°<br>Dec: %{lat:.2f}°<extra></extra>",
        )
    )

    # Add FOV circles around selected pixels
    for i, (ra_center, dec_center) in enumerate(zip(ra_selected, dec_selected)):
        # Convert RA to -180 to 180 range
        ra_center_plot = ra_center if ra_center <= 180 else ra_center - 360

        # Create circle points
        circle_points = _create_fov_circle(ra_center_plot, dec_center, fov_radius_deg)

        if circle_points is not None:
            fig.add_trace(
                go.Scattergeo(
                    lon=circle_points[:, 0],
                    lat=circle_points[:, 1],
                    mode="lines",
                    line=dict(width=1, color="rgba(255, 0, 0, 0.3)"),
                    name="FOV Circle" if i == 0 else None,
                    showlegend=(i == 0),
                    hoverinfo="skip",
                )
            )

    # Configure layout
    fig.update_layout(
        title="Survey Footprint with FOV Circles",
        geo=dict(
            projection_type="mollweide",
            showland=False,
            showocean=False,
            showcoastlines=False,
            showframe=True,
            framecolor="black",
            framewidth=1,
            bgcolor="white",
            lonaxis=dict(
                range=[-180, 180],
                dtick=30,
                showgrid=True,
                gridcolor="lightgray",
                gridwidth=0.5,
            ),
            lataxis=dict(
                range=[-90, 90],
                dtick=30,
                showgrid=True,
                gridcolor="lightgray",
                gridwidth=0.5,
            ),
        ),
        width=1000,
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def plot_survey_pointings(
    survey_pointings: SurveyPointings,
    survey: Survey,
    bundle_time_minutes: Optional[float] = None,
    show_fov_circles: bool = True,
    show_exposure_order: bool = True,
    width: int = 1200,
    height: int = 700,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Visualize survey pointings in Mollweide projection with time slider.

    Goes exposure by exposure, with optional time bundling to group nearby exposures.
    Shows FOV circles based on the survey's observatory configuration.

    Parameters
    ----------
    survey_pointings : SurveyPointings
        Table of survey pointings with coordinates and times
    survey : Survey
        Survey configuration object (used for FOV radius and metadata)
    bundle_time_minutes : float, optional
        If provided, bundle exposures within this time window (minutes).
        If None, show each exposure individually.
    show_fov_circles : bool, default True
        Whether to show field-of-view circles around each pointing
    show_exposure_order : bool, default True
        Whether to show exposure numbers/order
    width, height : int
        Figure dimensions in pixels
    title : str, optional
        Plot title. If None, generates automatic title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure with time slider
    """

    if len(survey_pointings) == 0:
        raise ValueError("No survey pointings provided")

    # Extract pointing data and convert to numpy arrays
    exposure_ids = survey_pointings.exposure_id.to_pylist()
    ra = survey_pointings.field_ra.to_numpy()
    dec = survey_pointings.field_dec.to_numpy()
    filters = survey_pointings.filter.to_pylist()
    exposure_times_mjd = survey_pointings.exposure_start.mjd()

    # Convert PyArrow arrays to numpy
    if hasattr(exposure_times_mjd, "to_numpy"):
        exposure_times_mjd = exposure_times_mjd.to_numpy()
    else:
        exposure_times_mjd = np.array(exposure_times_mjd)

    # Convert RA to -180 to 180 for Mollweide projection
    ra_plot = np.where(ra > 180, ra - 360, ra)

    # Get FOV radius from survey configuration
    fov_radius_deg = 1.5  # Default
    try:
        # Try to get from survey's observatory (if it has one)
        if hasattr(survey, "observatory") and survey.observatory is not None:
            if (
                survey.observatory.fov.camera_model == "circle"
                and survey.observatory.fov.circle_radius is not None
            ):
                fov_radius_deg = survey.observatory.fov.circle_radius
    except AttributeError:
        pass  # Use default

    # Create time groups
    if bundle_time_minutes is None:
        # Individual exposures
        time_groups = _create_individual_exposure_groups(
            exposure_ids,
            ra_plot,
            dec,
            filters,
            exposure_times_mjd,
        )
    else:
        # Bundle by time
        time_groups = _create_time_bundled_groups(
            exposure_ids,
            ra_plot,
            dec,
            filters,
            exposure_times_mjd,
            bundle_time_minutes,
        )

    if not time_groups:
        raise ValueError("No time groups created from survey pointings")

    # Create figure
    fig = go.Figure()

    # Create frames for each time step
    frames = []
    slider_steps = []

    for step_idx, (step_name, step_data) in enumerate(time_groups.items()):
        frame_traces = []

        # Add current exposures
        frame_traces.append(
            go.Scattergeo(
                lon=step_data["ra"],
                lat=step_data["dec"],
                mode="markers+text" if show_exposure_order else "markers",
                marker=dict(
                    size=12,
                    color=_get_filter_colors(step_data["filters"]),
                    opacity=0.8,
                    symbol="cross",
                    line=dict(width=2, color="black"),
                ),
                text=step_data["labels"] if show_exposure_order else None,
                textposition="top center",
                textfont=dict(size=10, color="black"),
                name="Current Exposures",
                customdata=list(
                    zip(
                        step_data["exposure_ids"],
                        step_data["filters"],
                        step_data["times_str"],
                    )
                ),
                hovertemplate=(
                    "Exposure: %{customdata[0]}<br>"
                    "Filter: %{customdata[1]}<br>"
                    "Time: %{customdata[2]}<br>"
                    "RA: %{lon:.3f}°<br>"
                    "Dec: %{lat:.3f}°<br>"
                    "<extra></extra>"
                ),
                showlegend=True,
            )
        )

        # Add FOV circles if requested
        if show_fov_circles:
            for ra_center, dec_center in zip(step_data["ra"], step_data["dec"]):
                circle_points = _create_fov_circle(
                    ra_center, dec_center, fov_radius_deg
                )
                if circle_points is not None:
                    frame_traces.append(
                        go.Scattergeo(
                            lon=circle_points[:, 0],
                            lat=circle_points[:, 1],
                            mode="lines",
                            line=dict(width=2, color="rgba(255, 0, 0, 0.6)"),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

        # Create frame
        frames.append(
            go.Frame(
                data=frame_traces,
                name=str(step_idx),
                layout=go.Layout(title=f"{title or 'Survey Pointings'} - {step_name}"),
            )
        )

        # Create slider step
        slider_steps.append(
            {
                "args": [
                    [str(step_idx)],
                    {
                        "frame": {"duration": 800, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300},
                    },
                ],
                "label": step_name,
                "method": "animate",
            }
        )

    # Add initial traces to figure (from first frame)
    if frames:
        for trace in frames[0].data:
            fig.add_trace(trace)

    # Add frames to figure
    fig.frames = frames

    # Configure layout
    fig.update_layout(
        title=dict(
            text=title or f"Survey Pointings - {survey.observatory_code}",
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
        geo=dict(
            projection_type="mollweide",
            showland=False,
            showocean=False,
            showcoastlines=False,
            showframe=True,
            framecolor="black",
            framewidth=1,
            bgcolor="white",
            lonaxis=dict(
                range=[-180, 180],
                dtick=30,
                showgrid=True,
                gridcolor="lightgray",
                gridwidth=0.5,
            ),
            lataxis=dict(
                range=[-90, 90],
                dtick=30,
                showgrid=True,
                gridcolor="lightgray",
                gridwidth=0.5,
            ),
        ),
        width=width,
        height=height,
        margin=dict(l=50, r=50, t=100, b=120),
        # Animation controls
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 1000, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 300},
                            },
                        ],
                        "label": "▶ Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "⏸ Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        # Time slider
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 14},
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 300},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": slider_steps,
            }
        ],
    )

    # Add metadata annotation
    fig.add_annotation(
        text=(
            f"Observatory: {survey.observatory_code} | "
            f"FOV: {fov_radius_deg:.1f}° | "
            f"Exposures: {len(survey_pointings)} | "
            f"Time Steps: {len(time_groups)}"
        ),
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.12,
        xanchor="center",
        yanchor="top",
        showarrow=False,
        font=dict(size=12, color="gray"),
    )

    return fig


def plot_night_sky_evolution(
    survey_footprint: SurveyFootprint,
    observatory_code: str,
    night_mjd: float,
    observing_start_time: float = 18.0,
    observing_duration_hours: float = 10.0,
    time_step_minutes: float = 30,
    max_zenith_angle: float = 70.0,
    animate: bool = True,
    height: int = 800,
    width: int = 1200,
    show_grid: bool = True,
    color_by_zenith: bool = True,
) -> go.Figure:
    """
    Create an animated visualization showing how the visible sky changes throughout a night.

    Parameters
    ----------
    survey_footprint : SurveyFootprint
        The HEALPix sky footprint to visualize
    observatory_code : str
        Observatory code (e.g., "X05", "W84")
    night_mjd : float
        MJD of the night to visualize
    local_sunset_hour : float
        Local hour when observing begins (24-hour format)
    night_duration_hours : float
        Duration of observing session in hours
    time_step_minutes : float
        Time step between animation frames in minutes
    max_zenith_angle : float
        Maximum zenith angle for visibility (degrees)
    animate : bool
        If True, create animation; if False, show final state only
    height, width : int
        Figure dimensions in pixels
    show_grid : bool
        Whether to show coordinate grid
    color_by_zenith : bool
        If True, color pixels by zenith angle; if False, uniform color

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure with Mollweide projection
    """

    # Calculate time steps throughout the night
    total_minutes = observing_duration_hours * 60
    time_offsets_minutes = np.arange(
        0, total_minutes + time_step_minutes, time_step_minutes
    )
    time_offsets_days = time_offsets_minutes / (60 * 24)  # Convert to fractional days

    # Adjust for local sunset time (convert to UTC offset)
    # This is a simplified approach - you might want to use proper timezone handling
    sunset_offset = (observing_start_time - 12) / 24  # Rough UTC offset
    observation_times_mjd = night_mjd + sunset_offset + time_offsets_days

    # Helper function to filter footprint by zenith angle
    def get_visible_footprint_with_zenith(mjd_time):
        time_stamp = Timestamp.from_mjd([mjd_time], scale="utc")
        observatory = Observers.from_code(observatory_code, time_stamp)

        # Get topocentric unit vector
        topocentric_unit_vector = transform_coordinates(
            observatory.coordinates,
            frame_out="equatorial",
            origin_out=OriginCodes.EARTH,
            representation_out=CartesianCoordinates,
        ).r_hat[0]

        # Calculate zenith distances for all pixels
        zenith_distances = np.degrees(
            np.arccos(np.dot(survey_footprint.r, topocentric_unit_vector))
        )

        # Create visibility mask
        visible_mask = zenith_distances <= max_zenith_angle

        if not np.any(visible_mask):
            return None, None, None

        # Get visible coordinates
        visible_ra = survey_footprint.ra.to_numpy()[visible_mask]
        visible_dec = survey_footprint.dec.to_numpy()[visible_mask]
        visible_zenith = zenith_distances[visible_mask]

        # Convert RA/Dec to longitude/latitude for Mollweide projection
        lon = (180.0 - visible_ra) % 360.0 - 180.0
        lat = visible_dec

        return lon, lat, visible_zenith

    if not animate:
        # Static visualization showing final state
        final_time = observation_times_mjd[-1]
        lon, lat, zenith = get_visible_footprint_with_zenith(final_time)

        if lon is None:
            # No visible sky
            fig = go.Figure()
            fig.add_annotation(
                text="No visible sky at this time and zenith angle limit",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16),
            )
        else:
            # Create color scale based on zenith angle if requested
            if color_by_zenith:
                colors = zenith
                colorscale = "Viridis_r"  # Reversed viridis (dark = low zenith = good)
                colorbar_title = "Zenith Angle (°)"
            else:
                colors = "blue"
                colorscale = None
                colorbar_title = None

            fig = go.Figure(
                data=go.Scattergeo(
                    lon=lon,
                    lat=lat,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=colors,
                        colorscale=colorscale,
                        colorbar=dict(title=colorbar_title) if colorbar_title else None,
                        opacity=0.7,
                        line=dict(width=0.5, color="white"),
                    ),
                    hovertemplate=(
                        "RA: %{customdata[0]:.2f}°<br>"
                        "Dec: %{customdata[1]:.2f}°<br>"
                        "Zenith: %{customdata[2]:.1f}°<br>"
                        "<extra></extra>"
                    ),
                    customdata=(
                        np.column_stack(
                            [(180.0 - lon) % 360.0, lat, zenith]  # Convert back to RA
                        )
                        if lon is not None
                        else None
                    ),
                )
            )

        # Set layout
        fig.update_layout(
            title=f"Visible Sky - {observatory_code} - MJD {final_time:.3f}<br>"
            f"<sub>Max zenith angle: {max_zenith_angle}°</sub>",
            geo=dict(
                projection=dict(type="mollweide"),
                showcountries=False,
                showcoastlines=False,
                showland=False,
                showocean=False,
                bgcolor="black",
                lataxis=dict(
                    showgrid=show_grid, gridwidth=0.5, gridcolor="gray", range=[-90, 90]
                ),
                lonaxis=dict(
                    showgrid=show_grid,
                    gridwidth=0.5,
                    gridcolor="gray",
                    range=[-180, 180],
                ),
            ),
            height=height,
            width=width,
            paper_bgcolor="black",
            font=dict(color="white"),
        )

        return fig

    else:
        # Animated visualization
        frames = []
        initial_data = None

        for i, mjd_time in enumerate(observation_times_mjd):
            lon, lat, zenith = get_visible_footprint_with_zenith(mjd_time)

            # Calculate local time for display
            local_hour = (observing_start_time + time_offsets_minutes[i] / 60) % 24

            if lon is None:
                # No visible sky - create empty frame
                frame_data = go.Scattergeo(
                    lon=[],
                    lat=[],
                    mode="markers",
                    marker=dict(size=8, color="blue", opacity=0.7),
                    hovertemplate="<extra></extra>",
                )
            else:
                # Create color scale
                if color_by_zenith:
                    colors = zenith
                    colorscale = "Viridis_r"
                    colorbar = dict(title="Zenith Angle (°)")
                else:
                    colors = "blue"
                    colorscale = None
                    colorbar = None

                frame_data = go.Scattergeo(
                    lon=lon,
                    lat=lat,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=colors,
                        colorscale=colorscale,
                        colorbar=colorbar,
                        opacity=0.7,
                        line=dict(width=0.5, color="white"),
                        cmin=0,  # Fix color scale range
                        cmax=max_zenith_angle,
                    ),
                    hovertemplate=(
                        "RA: %{customdata[0]:.2f}°<br>"
                        "Dec: %{customdata[1]:.2f}°<br>"
                        "Zenith: %{customdata[2]:.1f}°<br>"
                        "<extra></extra>"
                    ),
                    customdata=np.column_stack(
                        [(180.0 - lon) % 360.0, lat, zenith]  # Convert back to RA
                    ),
                )

            # Store initial data for base figure
            if i == 0:
                initial_data = frame_data

            # Create frame
            frame = go.Frame(
                data=[frame_data],
                name=str(i),
                layout=go.Layout(
                    title=f"Visible Sky - {observatory_code} - Local Time: {local_hour:.1f}h<br>"
                    f"<sub>MJD {mjd_time:.3f} | Max zenith: {max_zenith_angle}° | "
                    f"Visible pixels: {len(lon) if lon is not None else 0}</sub>"
                ),
            )
            frames.append(frame)

        # Create slider steps
        slider_steps = []
        for i, mjd_time in enumerate(observation_times_mjd):
            local_hour = (observing_start_time + time_offsets_minutes[i] / 60) % 24
            slider_steps.append(
                dict(
                    method="animate",
                    args=[
                        [str(i)],
                        dict(
                            frame=dict(duration=300, redraw=True),
                            transition=dict(duration=200),
                            mode="immediate",
                        ),
                    ],
                    label=f"{local_hour:.1f}h",
                )
            )

        # Create play/pause buttons
        updatemenus = [
            dict(
                type="buttons",
                showactive=False,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=500, redraw=True),
                                transition=dict(duration=200),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False), mode="immediate"
                            ),
                        ],
                    ),
                    dict(
                        label="🔄 Reset",
                        method="animate",
                        args=[
                            [str(0)],
                            dict(frame=dict(duration=0, redraw=True), mode="immediate"),
                        ],
                    ),
                ],
            )
        ]

        # Create the main figure
        fig = go.Figure(
            data=[initial_data] if initial_data else [],
            frames=frames,
            layout=go.Layout(
                title=f"Night Sky Evolution - {observatory_code}<br>"
                f"<sub>Night of MJD {night_mjd:.1f} | Duration: {observing_duration_hours}h | "
                f"Max zenith: {max_zenith_angle}°</sub>",
                geo=dict(
                    projection=dict(type="mollweide"),
                    showcountries=False,
                    showcoastlines=False,
                    showland=False,
                    showocean=False,
                    bgcolor="black",
                    lataxis=dict(
                        showgrid=show_grid,
                        gridwidth=0.5,
                        gridcolor="gray",
                        range=[-90, 90],
                    ),
                    lonaxis=dict(
                        showgrid=show_grid,
                        gridwidth=0.5,
                        gridcolor="gray",
                        range=[-180, 180],
                    ),
                ),
                height=height,
                width=width,
                paper_bgcolor="black",
                font=dict(color="white"),
                sliders=[
                    dict(
                        active=0,
                        steps=slider_steps,
                        x=0.1,
                        xanchor="left",
                        y=0.02,
                        yanchor="bottom",
                        len=0.8,
                        ticklen=0,
                        font=dict(size=10),
                    )
                ],
                updatemenus=updatemenus,
            ),
        )

        return fig
