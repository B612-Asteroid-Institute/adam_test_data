import multiprocessing as mp
import os
import sqlite3 as sql
import subprocess
from typing import Literal, Optional, Type, Union

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import quivr as qv
import ray
from adam_core.propagator.utils import _iterate_chunk_indices
from adam_core.ray_cluster import initialize_use_ray

from .observatory import Observatory, observatory_to_sorcha_config
from .pointings import Pointings
from .populations import (
    SmallBodies,
    orbits_to_sorcha_table,
    photometric_properties_to_sorcha_table,
)


class SorchaOutputBasic(qv.Table):
    # TODO: It would be really nice if the basic outputs included the FieldID
    # so that we could match the observations to the pointings
    ObjID = qv.LargeStringColumn()
    fieldMJD_TAI = qv.Float64Column()
    fieldRA_deg = qv.Float64Column()
    fieldDec_deg = qv.Float64Column()
    RA_deg = qv.Float64Column()
    Dec_deg = qv.Float64Column()
    astrometricSigma_deg = qv.Float64Column()
    optFilter = qv.LargeStringColumn()
    trailedSourceMag = qv.Float64Column()
    trailedSourceMagSigma = qv.Float64Column()
    fiveSigmaDepth_mag = qv.Float64Column()
    phase_deg = qv.Float64Column()
    Range_LTC_km = qv.Float64Column()
    RangeRate_LTC_km_s = qv.Float64Column()
    Obj_Sun_LTC_km = qv.Float64Column()


class SorchaOutputAll(qv.Table):

    # Some of the commented-out columns can be retroactively
    # calculated by joining with inputs to sorcha.

    ObjID = qv.LargeStringColumn()
    FieldID = qv.LargeStringColumn()
    fieldMJD_TAI = qv.Float64Column()
    fieldJD_TDB = qv.Float64Column()
    Range_LTC_km = qv.Float64Column()
    RangeRate_LTC_km_s = qv.Float64Column()
    RATrue_deg = qv.Float64Column()
    RARateCosDec_deg_day = qv.Float64Column()
    DecTrue_deg = qv.Float64Column()
    DecRate_deg_day = qv.Float64Column()
    # Obj_Sun_x_LTC_km = qv.Float64Column()
    # Obj_Sun_y_LTC_km = qv.Float64Column()
    # Obj_Sun_z_LTC_km = qv.Float64Column()
    # Obj_Sun_vx_LTC_km_s = qv.Float64Column()
    # Obj_Sun_vy_LTC_km_s = qv.Float64Column()
    # Obj_Sun_vz_LTC_km_s = qv.Float64Column()
    # Obs_Sun_x_km = qv.Float64Column()
    # Obs_Sun_y_km = qv.Float64Column()
    # Obs_Sun_z_km = qv.Float64Column()
    # Obs_Sun_vx_km_s = qv.Float64Column()
    # Obs_Sun_vy_km_s = qv.Float64Column()
    # Obs_Sun_vz_km_s = qv.Float64Column()
    phase_deg = qv.Float64Column()
    # FORMAT = qv.LargeStringColumn()
    # x = qv.Float64Column()
    # y = qv.Float64Column()
    # z = qv.Float64Column()
    # xdot = qv.Float64Column()
    # ydot = qv.Float64Column()
    # zdot = qv.Float64Column()
    # epochMJD_TDB = qv.Float64Column()
    # H_filter = qv.Float64Column()
    # VR_r = qv.Float64Column()
    # GS = qv.Float64Column()
    # visitTime = qv.Float64Column()
    # visitExposureTime = qv.Float64Column()
    optFilter = qv.LargeStringColumn()
    # seeingFwhmGeom_arcsec = qv.Float64Column()
    # seeingFwhmEff_arcsec = qv.Float64Column()
    fieldFiveSigmaDepth_mag = qv.Float64Column()
    # fieldRA_deg = qv.Float64Column()
    # fieldDec_deg = qv.Float64Column()
    # rotSkyPos_deg = qv.Float64Column()
    # observatory_code = qv.LargeStringColumn()
    # H_r = qv.Float64Column()
    trailedSourceMagTrue = qv.Float64Column()
    PSFMagTrue = qv.Float64Column()
    fiveSigmaDepth_mag = qv.Float64Column()
    astrometricSigma_deg = qv.Float64Column()
    trailedSourceMagSigma = qv.Float64Column()
    SNR = qv.Float64Column()
    PSFMagSigma = qv.Float64Column()
    trailedSourceMag = qv.Float64Column()
    PSFMag = qv.Float64Column()
    RA_deg = qv.Float64Column()
    Dec_deg = qv.Float64Column()
    Obj_Sun_LTC_km = qv.Float64Column()


class SorchaOutputStats(qv.Table):

    ObjID = qv.LargeStringColumn()
    optFilter = qv.LargeStringColumn()
    number_obs = qv.Int64Column(nullable=True)
    min_apparent_mag = qv.Float64Column(nullable=True)
    max_apparent_mag = qv.Float64Column(nullable=True)
    median_apparent_mag = qv.Float64Column(nullable=True)
    min_phase = qv.Float64Column(nullable=True)
    max_phase = qv.Float64Column(nullable=True)


def write_sorcha_inputs(
    output_dir: str,
    small_bodies: SmallBodies,
    pointings: Pointings,
    observatory: Observatory,
    time_range: Optional[list[float]] = None,
    format: Literal["csv", "whitespace"] = "csv",
    element_type: Literal["cartesian", "keplerian", "cometary"] = "cartesian",
    orbits_file_name: str = "orbits.csv",
    properties_file_name: str = "properties.csv",
    pointings_database_name: str = "pointings.db",
    sorcha_config_file_name: str = "config.ini",
    randomization: bool = True,
    output_columns: Literal["basic", "all"] = "all",
) -> dict[str, str]:
    """
    Write the small body population, pointings, and observatory to files that can be read by Sorcha.

    Parameters
    ----------
    output_dir : str
        The directory to write the files to.
    small_bodies : SmallBodies
        The small body population to write.
    pointings : Pointings
        The pointings to write to a SQLite database.
    observatory : Observatory
        The observatory to write to a configuration file.
    time_range : list[float], optional
        The time range to filter the pointings by, by default None.
    format : Literal["csv", "whitespace"], optional
        The format to write the orbits and photometric properties in, by default "csv".
    element_type : Literal["cartesian", "keplerian", "cometary"], optional
        The type of orbital elements to write, by default "cartesian".
    orbits_file_name : str, optional
        The name of the file to write the orbits to, by default "orbits.csv".
    properties_file_name : str, optional
        The name of the file to write the photometric properties to, by default "properties.csv".
    pointings_database_name : str, optional
        The name of the SQLite database to write the pointings to, by default "pointings.db".
    sorcha_config_file_name : str, optional
        The name of the configuration file to write the observatory to, by default "config.ini".
    randomization : bool, optional
        Ramdomize the photometry and astrometry using the calculated uncertainties, by default True.
    output_columns : Literal["basic", "all"], optional
        The columns to output in the Sorcha output, by default "all".

    Returns
    -------
    paths : dict
        A dictionary containing the paths to the files with the following keys:
        - "orbits" : The path to the orbits file.
        - "photometric_properties" : The path to the photometric properties file.
        - "pointings" : The path to the pointings database.
        - "config" : The path to the configuration file.
    """
    paths = {}

    os.makedirs(output_dir, exist_ok=True)

    # Write orbits and photometric properties to the corresponding file types
    orbits_table = orbits_to_sorcha_table(small_bodies.orbits, element_type)
    properties_table = photometric_properties_to_sorcha_table(
        small_bodies.properties, small_bodies.orbits.object_id, observatory.main_filter
    )

    orbit_file = os.path.join(output_dir, orbits_file_name)
    properties_file = os.path.join(output_dir, properties_file_name)

    if format == "csv":
        delimiter = ","
    elif format == "whitespace":
        delimiter = " "
    else:
        raise ValueError("format must be either 'csv' or 'whitespace'")

    # Normally, it should be as easy as this:
    # pa.csv.write_csv(
    #     orbits_table,
    #     orbit_file,
    #     write_options=pa.csv.WriteOptions(
    #         include_header=True, delimiter=delimiter, quoting_style="none"
    #     ),
    # )
    # pa.csv.write_csv(
    #     properties_table,
    #     properties_file,
    #     write_options=pa.csv.WriteOptions(
    #         include_header=True,
    #         delimiter=delimiter,
    #         quoting_style="none",
    #     ),
    # )
    # But sorcha doesn't know how to handle quotes in the header
    # and the quoting_style="none" only applies to the row data and
    # not the header. That is, the header names will remain surrounded
    # by double quotes.
    # See: https://github.com/apache/arrow/issues/41239

    # Instead we write the CSVs to a string buffer
    # and then strip the quotes from the header.
    #
    # Nevermind, this also doesn't work because object IDs can have
    # "structural characters". Here is an example error if you
    # set quoting_style="none":
    # CSV values may not contain structural characters
    # if quoting style is "None". See RFC4180. Invalid value: (2013 RR165)
    #
    # Furthermore, sorcha can't handle quotes in the data either...
    # Instead we use the "needed" quoting style which adds quotes to the
    # row data and the header. We then strip the quotes from everything with
    # a replace call.
    with pa.BufferOutputStream() as stream:
        pa.csv.write_csv(
            orbits_table,
            stream,
            write_options=pa.csv.WriteOptions(
                include_header=True, delimiter=delimiter, quoting_style="needed"
            ),
        )
        orbits_csv = stream.getvalue().to_pybytes().decode("utf-8")

        # Strip the quotes from the header and body
        orbits_csv = orbits_csv.replace('"', "")

        with open(orbit_file, "w") as f:
            f.write(orbits_csv)

    with pa.BufferOutputStream() as stream:
        pa.csv.write_csv(
            properties_table,
            stream,
            write_options=pa.csv.WriteOptions(
                include_header=True,
                delimiter=delimiter,
                quoting_style="needed",
            ),
        )
        properties_csv = stream.getvalue().to_pybytes().decode("utf-8")

        # Strip the quotes from the header and body
        properties_csv = properties_csv.replace('"', "")

        with open(properties_file, "w") as f:
            f.write(properties_csv)

    paths["orbits"] = orbit_file
    paths["photometric_properties"] = properties_file

    # Write pointings to a sqlite database
    pointings_file = os.path.join(output_dir, pointings_database_name)
    con = sql.connect(pointings_file)
    pointings.to_sql(con, table_name="pointings")
    con.close()

    paths["pointings"] = pointings_file

    # Write the observatory to a configuration file
    config_file = os.path.join(output_dir, sorcha_config_file_name)
    with open(config_file, "w") as f:
        config_str = observatory_to_sorcha_config(
            observatory,
            time_range=time_range,
            randomization=randomization,
            output_columns=output_columns,
        )
        f.write(config_str)

    paths["config"] = config_file

    return paths


def sorcha(
    output_dir: str,
    small_bodies: SmallBodies,
    pointings: Pointings,
    observatory: Observatory,
    time_range: Optional[list[float]] = None,
    tag: str = "sorcha",
    overwrite: bool = True,
    randomization: bool = True,
    output_columns: Literal["basic", "all"] = "all",
) -> tuple[Union[SorchaOutputBasic, SorchaOutputAll], SorchaOutputStats]:
    """
    Run sorcha on the given small bodies, pointings, and observatory.

    Parameters
    ----------
    output_dir : str
        The directory to write the Sorcha output to.
    small_bodies : SmallBodies
        The small body population to run Sorcha on.
    pointings : Pointings
        The pointings to run Sorcha on.
    observatory : Observatory
        The observatory to run Sorcha on.
    time_range : list[float], optional
        The time range to filter the pointings by, by default None.
    tag : str, optional
        The tag to use for the output files, by default "sorcha".
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    randomization : bool, optional
        Ramdomize the photometry and astrometry using the calculated uncertainties, by default True.
    output_columns : Literal["basic", "all"], optional
        The columns to output in the Sorcha output, by default "all".

    Returns
    -------
    tuple[Union[SorchaOutputBasic, SorchaOutputAll], SorchaOutputStats]
        Sorcha output observations (in basic or all formats) and statistics
        per object and filter.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    paths = write_sorcha_inputs(
        output_dir,
        small_bodies,
        pointings,
        observatory,
        time_range=time_range,
        randomization=randomization,
        output_columns=output_columns,
    )

    # Note that the stats file is automatically saved in the output directory and with a
    # preset file extension so we do not need to add the output directory to the stats file path
    stats_file = f"{tag}_stats"
    command = [
        "sorcha",
        "run",
        "-c",
        f"{paths['config']}",
        "-p",
        f"{paths['photometric_properties']}",
        "-ob",
        f"{paths['orbits']}",
        "-pd",
        f"{paths['pointings']}",
        "-o",
        f"{output_dir}",
        "-t",
        f"{tag}",
        "-st",
        f"{stats_file}",
    ]
    if overwrite:
        command.append("-f")

    results = subprocess.run(command, check=False, capture_output=True)
    if results.returncode != 0:
        raise RuntimeError(
            f"Sorcha failed with the following error:\n{results.stderr.decode()}"
        )
    else:
        output_file = f"{output_dir}/{tag}.csv"
        output_stats_file = f"{output_dir}/{stats_file}.csv"

        sorcha_output_table: Union[Type[SorchaOutputBasic], Type[SorchaOutputAll]]
        if output_columns == "basic":
            sorcha_output_table = SorchaOutputBasic
        else:
            sorcha_output_table = SorchaOutputAll

        # If no ephemerides are found by sorcha it will not generate
        # a csv file for either the simulated observations or
        # the statistics.
        sorcha_outputs: Union[SorchaOutputBasic, SorchaOutputAll]
        if os.path.exists(output_file):
            sorcha_outputs = sorcha_output_table.from_csv(output_file)
            sorcha_stats = SorchaOutputStats.from_csv(output_stats_file)
        else:
            sorcha_outputs = sorcha_output_table.empty()
            sorcha_stats = SorchaOutputStats.empty()

    return sorcha_outputs, sorcha_stats


def sorcha_worker(
    orbit_ids: pa.Array,
    orbit_ids_indices: tuple[int, int],
    output_dir: str,
    small_bodies: SmallBodies,
    pointings: Pointings,
    observatory: Observatory,
    time_range: Optional[list[float]] = None,
    tag: str = "sorcha",
    overwrite: bool = True,
    randomization: bool = True,
    output_columns: Literal["basic", "all"] = "all",
    cleanup: bool = True,
) -> tuple[str, str]:
    """
    Run sorcha on a subset of the input small bodies.

    Parameters
    ----------
    orbit_ids : pa.Array
        The orbit IDs of the small bodies.
    orbit_ids_indices : tuple[int, int]
        The indices of the orbit IDs to process.
    output_dir : str
        The directory to write the Sorcha output to.
    small_bodies : SmallBodies
        The small body population to run Sorcha on.
    pointings : Pointings
        The pointings to run Sorcha on.
    observatory : Observatory
        The observatory to run Sorcha on.
    time_range : list[float], optional
        The time range to filter the pointings by, by default None.
    tag : str, optional
        The tag to use for the output files, by default "sorcha".
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    randomization : bool, optional
        Ramdomize the photometry and astrometry using the calculated uncertainties, by default True.
    output_columns : Literal["basic", "all"], optional
        The columns to output in the Sorcha output, by default "all".

    Returns
    -------
    tuple[str, str]
        The paths to the Sorcha output files.
    """
    orbit_ids_chunk = orbit_ids[orbit_ids_indices[0] : orbit_ids_indices[1]]

    small_bodies_chunk = small_bodies.apply_mask(
        pc.is_in(small_bodies.orbits.orbit_id, orbit_ids_chunk)
    )

    # Create a subdirectory for this chunk
    chunk_base = f"chunk_{orbit_ids_indices[0]:08d}_{orbit_ids_indices[1]:08d}"
    output_dir_chunk = os.path.join(output_dir, chunk_base)

    sorcha_outputs, sorcha_stats = sorcha(
        output_dir_chunk,
        small_bodies_chunk,
        pointings,
        observatory,
        time_range=time_range,
        tag=tag,
        overwrite=overwrite,
        randomization=randomization,
        output_columns=output_columns,
    )

    # Serialize the output tables to parquet and return the paths
    sorcha_output_file = os.path.join(output_dir, f"{chunk_base}_{tag}.parquet")
    sorcha_stats_file = os.path.join(output_dir, f"{chunk_base}_{tag}_stats.parquet")

    sorcha_outputs.to_parquet(sorcha_output_file)
    sorcha_stats.to_parquet(sorcha_stats_file)

    return sorcha_output_file, sorcha_stats_file


sorcha_worker_remote = ray.remote(sorcha_worker)
sorcha_worker_remote.options(num_cpus=1)


def run_sorcha(
    output_dir: str,
    small_bodies: SmallBodies,
    pointings: Pointings,
    observatory: Observatory,
    time_range: Optional[list[float]] = None,
    tag: str = "sorcha",
    overwrite: bool = True,
    randomization: bool = True,
    output_columns: Literal["basic", "all"] = "all",
    chunk_size: int = 1000,
    max_processes: Optional[int] = 1,
) -> tuple[str, str]:
    """
    Run sorcha on the given small bodies, pointings, and observatory.

    Parameters
    ----------
    output_dir : str
        The directory to write the Sorcha output to.
    small_bodies : SmallBodies
        The small body population to run Sorcha on.
    pointings : Pointings
        The pointings to run Sorcha on.
    observatory : Observatory
        The observatory to run Sorcha on.
    time_range : list[float], optional
        The time range to filter the pointings by, by default None.
    tag : str, optional
        The tag to use for the output files, by default "sorcha".
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    randomization : bool, optional
        Ramdomize the photometry and astrometry using the calculated uncertainties, by default True.
    output_columns : Literal["basic", "all"], optional
        The columns to output in the Sorcha output, by default "all".
    chunk_size : int, optional
        The number of small bodies to process in each chunk, by default 1000.
    max_processes : Optional[int], optional
        The maximum number of processes to use, by default 1.

    Returns
    -------
    tuple[Union[SorchaOutputBasic, SorchaOutputAll], SorchaOutputStats]
        Sorcha output observations (in basic or all formats) and statistics
        per object and filter.
    """
    if max_processes is None:
        max_processes = mp.cpu_count()

    orbit_ids = small_bodies.orbits.orbit_id

    sorcha_outputs: Union[SorchaOutputBasic, SorchaOutputAll]
    if output_columns == "basic":
        sorcha_outputs = SorchaOutputBasic.empty()
    else:
        sorcha_outputs = SorchaOutputAll.empty()
    sorcha_stats = SorchaOutputStats.empty()

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    sorcha_output_file = os.path.join(output_dir, f"{tag}.parquet")
    sorcha_stats_file = os.path.join(output_dir, f"{tag}_stats.parquet")

    # Write the empty tables to parquet
    sorcha_outputs.to_parquet(sorcha_output_file)
    sorcha_stats.to_parquet(sorcha_stats_file)

    sorcha_outputs_writer = pq.ParquetWriter(
        sorcha_output_file,
        sorcha_outputs.schema,
    )
    sorcha_stats_writer = pq.ParquetWriter(
        sorcha_stats_file,
        sorcha_stats.schema,
    )

    use_ray = initialize_use_ray(num_cpus=max_processes)
    if use_ray:

        orbit_ids_ref = ray.put(orbit_ids)
        small_bodies_ref = ray.put(small_bodies)
        pointings_ref = ray.put(pointings)
        observatory_ref = ray.put(observatory)

        futures = []
        for orbit_ids_indices in _iterate_chunk_indices(orbit_ids, chunk_size):
            futures.append(
                sorcha_worker_remote.remote(
                    orbit_ids_ref,
                    orbit_ids_indices,
                    output_dir,
                    small_bodies_ref,
                    pointings_ref,
                    observatory=observatory_ref,
                    time_range=time_range,
                    tag=tag,
                    overwrite=overwrite,
                    randomization=randomization,
                    output_columns=output_columns,
                )
            )

            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                sorcha_outputs_chunk_file, sorcha_stats_chunk_file = ray.get(
                    finished[0]
                )

                sorcha_outputs_chunk = sorcha_outputs.from_parquet(
                    sorcha_outputs_chunk_file
                )
                sorcha_stats_chunk = sorcha_stats.from_parquet(sorcha_stats_chunk_file)

                sorcha_outputs_writer.write_table(sorcha_outputs_chunk.table)
                sorcha_stats_writer.write_table(sorcha_stats_chunk.table)

                os.remove(sorcha_outputs_chunk_file)
                os.remove(sorcha_stats_chunk_file)

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            sorcha_outputs_chunk_file, sorcha_stats_chunk_file = ray.get(finished[0])

            sorcha_outputs_chunk = sorcha_outputs.from_parquet(
                sorcha_outputs_chunk_file
            )
            sorcha_stats_chunk = sorcha_stats.from_parquet(sorcha_stats_chunk_file)

            sorcha_outputs_writer.write_table(sorcha_outputs_chunk.table)
            sorcha_stats_writer.write_table(sorcha_stats_chunk.table)

            os.remove(sorcha_outputs_chunk_file)
            os.remove(sorcha_stats_chunk_file)

    else:

        for orbit_ids_indices in _iterate_chunk_indices(orbit_ids, chunk_size):
            sorcha_outputs_chunk_file, sorcha_stats_chunk_file = sorcha_worker(
                orbit_ids,
                orbit_ids_indices,
                output_dir,
                small_bodies,
                pointings,
                observatory,
                time_range=time_range,
                tag=tag,
                overwrite=overwrite,
                randomization=randomization,
                output_columns=output_columns,
            )

            sorcha_outputs_chunk = sorcha_outputs.from_parquet(
                sorcha_outputs_chunk_file
            )
            sorcha_stats_chunk = sorcha_stats.from_parquet(sorcha_stats_chunk_file)

            sorcha_outputs_writer.write_table(sorcha_outputs_chunk.table)
            sorcha_stats_writer.write_table(sorcha_stats_chunk.table)

            os.remove(sorcha_outputs_chunk_file)
            os.remove(sorcha_stats_chunk_file)

    return sorcha_output_file, sorcha_stats_file
