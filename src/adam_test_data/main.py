import multiprocessing as mp
import os
import shutil
import sqlite3 as sql
import subprocess
import uuid
from abc import ABC, abstractmethod
from typing import Literal, Optional, Type, Union

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import quivr as qv
import ray
from adam_core.observations import SourceCatalog
from adam_core.propagator.utils import _iterate_chunk_indices
from adam_core.ray_cluster import initialize_use_ray
from adam_core.time import Timestamp

from .noise import generate_noise
from .observatories import Observatory, observatory_to_sorcha_config
from .pointings import Pointings
from .populations import (
    SmallBodies,
    orbits_to_sorcha_table,
    photometric_properties_to_sorcha_table,
)


class SorchaDerivedOutputs(ABC):

    @abstractmethod
    def to_source_catalog(
        self, catalog_id: str, exposure_id: str, observatory_code: str
    ) -> SourceCatalog:
        """
        Convert the Sorcha output to a SourceCatalog.

        Parameters
        ----------
        catalog_id : str
            The ID of the catalog.
        exposure_id : str
            The ID of the exposure.
        observatory_code : str
            The code of the observatory.

        Returns
        -------
        source_catalog : SourceCatalog
            The source catalog.
        """
        pass


class SorchaOutputBasic(qv.Table, SorchaDerivedOutputs):
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

    def to_source_catalog(
        self, catalog_id: str, exposure_id: str, observatory_code: str
    ) -> SourceCatalog:
        """
        Convert the Sorcha output to a SourceCatalog.

        Parameters
        ----------
        catalog_id : str
            The ID of the catalog.
        exposure_id : str
            The ID of the exposure.
        observatory_code : str
            The code of the observatory.

        Returns
        -------
        source_catalog : SourceCatalog
            The source catalog.
        """
        num_obs = len(self)
        obs_ids = [uuid.uuid4().hex for _ in range(num_obs)]
        catalog_id = pa.repeat(catalog_id, num_obs)
        exposure_id_arr = pa.repeat(exposure_id, num_obs)
        observatory_code_arr = pa.repeat(observatory_code, num_obs)

        return SourceCatalog.from_kwargs(
            id=obs_ids,
            exposure_id=exposure_id_arr,
            time=Timestamp.from_mjd(
                self.fieldMJD_TAI,
                scale="tai",
            ),
            ra=self.RA_deg,
            dec=self.Dec_deg,
            ra_sigma=self.astrometricSigma_deg,
            dec_sigma=self.astrometricSigma_deg,
            mag=self.trailedSourceMag,
            mag_sigma=self.trailedSourceMagSigma,
            observatory_code=observatory_code_arr,
            filter=self.optFilter,
            exposure_start_time=Timestamp.from_mjd(
                self.fieldMJD_TAI,
                scale="tai",
            ),
            object_id=self.ObjID,
            catalog_id=catalog_id,
        )


class SorchaOutputAll(qv.Table, SorchaDerivedOutputs):

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
    Obj_Sun_x_LTC_km = qv.Float64Column()
    Obj_Sun_y_LTC_km = qv.Float64Column()
    Obj_Sun_z_LTC_km = qv.Float64Column()
    Obj_Sun_vx_LTC_km_s = qv.Float64Column()
    Obj_Sun_vy_LTC_km_s = qv.Float64Column()
    Obj_Sun_vz_LTC_km_s = qv.Float64Column()
    Obs_Sun_x_km = qv.Float64Column()
    Obs_Sun_y_km = qv.Float64Column()
    Obs_Sun_z_km = qv.Float64Column()
    Obs_Sun_vx_km_s = qv.Float64Column()
    Obs_Sun_vy_km_s = qv.Float64Column()
    Obs_Sun_vz_km_s = qv.Float64Column()
    phase_deg = qv.Float64Column()
    FORMAT = qv.LargeStringColumn()
    x = qv.Float64Column()
    y = qv.Float64Column()
    z = qv.Float64Column()
    xdot = qv.Float64Column()
    ydot = qv.Float64Column()
    zdot = qv.Float64Column()
    epochMJD_TDB = qv.Float64Column()
    H_filter = qv.Float64Column()
    H_r = qv.Float64Column()
    GS = qv.Float64Column()
    visitTime = qv.Float64Column()
    visitExposureTime = qv.Float64Column()
    optFilter = qv.LargeStringColumn()
    seeingFwhmGeom_arcsec = qv.Float64Column()
    seeingFwhmEff_arcsec = qv.Float64Column()
    fieldFiveSigmaDepth_mag = qv.Float64Column()
    fieldRA_deg = qv.Float64Column()
    fieldDec_deg = qv.Float64Column()
    rotSkyPos_deg = qv.Float64Column()
    observatory_code = qv.LargeStringColumn()
    H_r = qv.Float64Column()
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

    def to_source_catalog(
        self, catalog_id: str, exposure_id: str, observatory_code: str
    ) -> SourceCatalog:
        """
        Convert the Sorcha output to a SourceCatalog.

        Parameters
        ----------
        catalog_id : str
            The ID of the catalog.
        exposure_id : str
            The ID of the exposure.
        observatory_code : str
            The code of the observatory (unused here
            and directly read from the Sorcha output).

        Returns
        -------
        source_catalog : SourceCatalog
            The source catalog.
        """
        num_obs = len(self)
        obs_ids = [uuid.uuid4().hex for _ in range(num_obs)]
        catalog_id = pa.repeat(catalog_id, num_obs)
        return SourceCatalog.from_kwargs(
            id=obs_ids,
            exposure_id=self.FieldID,
            time=Timestamp.from_mjd(
                self.fieldMJD_TAI,
                scale="tai",
            ),
            ra=self.RA_deg,
            dec=self.Dec_deg,
            ra_sigma=self.astrometricSigma_deg,
            dec_sigma=self.astrometricSigma_deg,
            # Here we use the trailed source mag as the mag.
            # In the limit where the source is not trailed, this should
            # approach the PSFMag.
            mag=self.trailedSourceMag,
            mag_sigma=self.trailedSourceMagSigma,
            observatory_code=self.observatory_code,
            filter=self.optFilter,
            exposure_start_time=Timestamp.from_mjd(
                self.fieldMJD_TAI,
                scale="tai",
            ),
            exposure_duration=self.visitExposureTime,
            # Here we use the seeingFwhmEff_arcsec as the seeing, this will
            # be a more conservative estimate of the seeing than the
            # seeingFwhmGeom_arcsec.
            # See: https://community.lsst.org/t/difference-between-seeingfwhmeff-and-seeingfwhmgeom/8137/2
            exposure_seeing=self.seeingFwhmEff_arcsec,
            exposure_depth_5sigma=self.fieldFiveSigmaDepth_mag,
            object_id=self.ObjID,
            catalog_id=catalog_id,
        )


class SorchaOutputStats(qv.Table):

    ObjID = qv.LargeStringColumn()
    optFilter = qv.LargeStringColumn()
    number_obs = qv.Int64Column(nullable=True)
    min_apparent_mag = qv.Float64Column(nullable=True)
    max_apparent_mag = qv.Float64Column(nullable=True)
    median_apparent_mag = qv.Float64Column(nullable=True)
    min_phase = qv.Float64Column(nullable=True)
    max_phase = qv.Float64Column(nullable=True)


class TestDataSummary(qv.Table):

    catalog_id = qv.StringColumn()
    start_time = Timestamp.as_column()
    end_time = Timestamp.as_column()
    num_orbits = qv.Int64Column()
    population_name = qv.StringColumn()
    pointings_name = qv.StringColumn()
    observatory_code = qv.StringColumn()
    noise_density = qv.Float64Column(nullable=True)
    catalog_file = qv.StringColumn()
    noise_file = qv.StringColumn(nullable=True)

    def load_test_data(
        self, catalog_id: str, noise_density: Optional[float] = None
    ) -> SourceCatalog:
        """
        Convenience method to load the test data for a given catalog_id and noise_density.

        Parameters
        ----------
        catalog_id : str
            The ID of the catalog.
        noise_density : float, optional
            The noise observations at a particular density
            to load with the catalog, by default None.

        Returns
        -------
        catalog : SourceCatalog
            The test data catalog (with noise if noise_density is not None).
        """
        catalog_summary = self.select("catalog_id", catalog_id)
        if len(catalog_summary) == 0:
            err = (
                f"No catalog found for catalog_id={catalog_id}\n"
                "Options are:\n"
                f"{self.catalog_id.unique().to_pylist()}"
            )
            raise ValueError(err)

        catalog = SourceCatalog.from_parquet(catalog_summary.catalog_file[0].as_py())

        if noise_density is not None:
            catalog_summary_noise = catalog_summary.select(
                "noise_density", noise_density
            )

            if len(catalog_summary_noise) == 0:
                err = (
                    f"No noise catalog found for catalog_id={catalog_id}"
                    f" and noise_density={noise_density}\n"
                    "Options are:\n"
                    f"{catalog_summary.noise_density.unique().to_pylist()}"
                )
                raise ValueError(err)

            noise_catalog = SourceCatalog.from_parquet(
                catalog_summary_noise.noise_file[0].as_py()
            )
            catalog = qv.concatenate([catalog, noise_catalog])

        return catalog


def remove_quotes(file_path: str) -> None:
    """
    Remove quotes from a CSV file.

    Parameters
    ----------
    file_path : str
    """
    # Create a temporary file path
    temp_file_path = file_path + ".tmp"

    # Open the original file for reading in binary mode
    # and the temporary file for writing in binary mode
    with open(file_path, "rb") as infile, open(temp_file_path, "wb") as outfile:
        while True:
            # Read a chunk of data (e.g., 64KB)
            chunk = infile.read(65536)
            if not chunk:
                break
            # Remove quote characters
            chunk = chunk.replace(b'"', b"")
            # Write the processed chunk to the temporary file
            outfile.write(chunk)

    # Replace the original file with the temporary file
    os.replace(temp_file_path, file_path)


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
        Ramdomize the photometry and astrometry using the calculated uncertainties.
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

    pa.csv.write_csv(
        orbits_table,
        orbit_file,
        write_options=pa.csv.WriteOptions(
            include_header=True, delimiter=delimiter, quoting_style="needed"
        ),
    )
    remove_quotes(orbit_file)

    pa.csv.write_csv(
        properties_table,
        properties_file,
        write_options=pa.csv.WriteOptions(
            include_header=True,
            delimiter=delimiter,
            quoting_style="needed",
        ),
    )
    remove_quotes(properties_file)

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
    cleanup: bool = True,
) -> SourceCatalog:
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
        Ramdomize the photometry and astrometry using the calculated uncertainties.
    output_columns : Literal["basic", "all"], optional
        The columns to output in the Sorcha output, by default "all".
    cleanup : bool, optional
        Whether to delete the input files and output files after running Sorcha.

    Returns
    -------
    SourceCatalog
        Sorcha outputs as an adam_core SourceCatalog.
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
            source_catalog = sorcha_outputs.to_source_catalog(
                catalog_id=tag,
                exposure_id=pointings.observationId[0].as_py(),
                observatory_code=observatory.code,
            )

        else:
            sorcha_outputs = sorcha_output_table.empty()
            source_catalog = SourceCatalog.empty()

    if cleanup:
        shutil.rmtree(output_dir)

    return source_catalog


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
) -> str:
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
        Ramdomize the photometry and astrometry using the calculated uncertainties.
    output_columns : Literal["basic", "all"], optional
        The columns to output in the Sorcha output, by default "all".

    Returns
    -------
    str
        The path to the SourceCatalog output file.
    """
    orbit_ids_chunk = orbit_ids[orbit_ids_indices[0] : orbit_ids_indices[1]]

    small_bodies_chunk = small_bodies.apply_mask(
        pc.is_in(small_bodies.orbits.orbit_id, orbit_ids_chunk)
    )

    # Create a subdirectory for this chunk
    chunk_base = f"{orbit_ids_indices[0]:08d}_{orbit_ids_indices[1]:08d}"
    output_dir_chunk = os.path.join(output_dir, chunk_base)

    catalog = sorcha(
        output_dir_chunk,
        small_bodies_chunk,
        pointings,
        observatory,
        time_range=time_range,
        tag=tag,
        overwrite=overwrite,
        randomization=randomization,
        output_columns=output_columns,
        cleanup=cleanup,
    )

    # Serialize the output tables to parquet and return the paths
    catalog_file = os.path.join(output_dir, f"{tag}_{chunk_base}.parquet")
    catalog.to_parquet(catalog_file)

    return catalog_file


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
    cleanup: bool = True,
) -> str:
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
        Ramdomize the photometry and astrometry using the calculated uncertainties.
    output_columns : Literal["basic", "all"], optional
        The columns to output in the Sorcha output, by default "all".
    chunk_size : int, optional
        The number of small bodies to process in each chunk, by default 1000.
    max_processes : Optional[int], optional
        The maximum number of processes to use, by default 1.
    cleanup : bool, optional
        Whether to delete the input files and output files after running Sorcha.

    Returns
    -------
    str
        The path to the SourceCatalog output file.
    """
    if max_processes is None:
        max_processes = mp.cpu_count()

    orbit_ids = small_bodies.orbits.orbit_id

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    catalog = SourceCatalog.empty()
    catalog_file = os.path.abspath(os.path.join(output_dir, f"{tag}.parquet"))
    catalog.to_parquet(catalog_file)

    # Create a Parquet writer for the output catalog
    catalog_writer = pq.ParquetWriter(
        catalog_file,
        catalog.schema,
    )

    use_ray = initialize_use_ray(num_cpus=max_processes)
    if use_ray:

        orbit_ids_ref = ray.put(orbit_ids)
        small_bodies_ref = ray.put(small_bodies)
        pointings_ref = ray.put(pointings)
        observatory_ref = ray.put(observatory)

        chunk_size = min(int((len(small_bodies) / max_processes)), chunk_size)
        chunk_size = max(1, int(chunk_size))

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
                    cleanup=cleanup,
                )
            )

            if len(futures) >= max_processes * 1.5:
                finished, futures = ray.wait(futures, num_returns=1)
                catalog_chunk_file = ray.get(finished[0])

                catalog_chunk = SourceCatalog.from_parquet(catalog_chunk_file)
                catalog_writer.write_table(catalog_chunk.table)

                if cleanup:
                    os.remove(catalog_chunk_file)

        while futures:
            finished, futures = ray.wait(futures, num_returns=1)
            catalog_chunk_file = ray.get(finished[0])

            catalog_chunk = SourceCatalog.from_parquet(catalog_chunk_file)
            catalog_writer.write_table(catalog_chunk.table)

            if cleanup:
                os.remove(catalog_chunk_file)

    else:

        for orbit_ids_indices in _iterate_chunk_indices(orbit_ids, chunk_size):
            catalog_chunk_file = sorcha_worker(
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
                cleanup=cleanup,
            )

            catalog_chunk = SourceCatalog.from_parquet(catalog_chunk_file)
            catalog_writer.write_table(catalog_chunk.table)

            if cleanup:
                os.remove(catalog_chunk_file)

    return catalog_file


def generate_test_data(
    output_dir: str,
    small_bodies: SmallBodies,
    pointings: Pointings,
    observatory: Observatory,
    noise_densities: Optional[list[float]] = None,
    time_range: Optional[list[float]] = None,
    tag: Optional[str] = None,
    overwrite: bool = True,
    randomization: bool = True,
    output_columns: Literal["basic", "all"] = "all",
    seed: Optional[int] = None,
    chunk_size: int = 1000,
    max_processes: Optional[int] = 1,
    cleanup: bool = True,
) -> tuple[str, dict[str, str], TestDataSummary]:
    """
    Generate a test data set optionally with noise observations.

    Parameters
    ----------
    output_dir : str
        The directory to write the test data to.
    small_bodies : SmallBodies
        The small body population to generate test data for.
    pointings : Pointings
        The pointings to generate test data for.
    observatory : Observatory
        The observatory to generate test data for.
    noise_densities : Optional[list[float]], optional
        The noise densities in detections per square degree to generate noise observations for, by default None.
        Each noise density will generate a separate set of noise observations stored
        in its own file.
    time_range : list[float], optional
        The time range to filter the pointings by, by default None.
    tag : str, optional
        The tag to use for the output files and the catalog ID. If None, the tag will be
        generated from the pointings name, observatory code, and small bodies name.
    randomization : bool, optional
        Ramdomize the photometry and astrometry using the calculated uncertainties.
    output_columns : Literal["basic", "all"], optional
        The columns to output in the Sorcha output, by default "all".
    seed : Optional[int], optional
        The seed to use for generating noise observations, by default None.
    chunk_size : int, optional
        The number of small bodies to process in each chunk, by default 1000.
        Also, the number of pointings to process in each chunk when
        generating noise observations.
    max_processes : Optional[int], optional
        The maximum number of processes to use, by default 1.
    cleanup : bool, optional
        Whether to delete the input files and output files after generating the test data.

    Returns
    -------
    tuple[str, dict[str, str]]
        The paths to the source catalog file, and a dictionary with paths to the noise files.
    """
    # Lets filter the pointings here first
    if time_range is not None:
        pointings_filtered = pointings.apply_mask(
            pc.and_(
                pc.greater_equal(pointings.observationStartMJD_TAI, time_range[0]),
                pc.less_equal(pointings.observationStartMJD_TAI, time_range[1]),
            )
        )
    else:
        pointings_filtered = pointings

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if tag is None:
        tag = f"{pointings.name}_{observatory.code}_{small_bodies.name}"

    # Run sorcha
    catalog_file = run_sorcha(
        output_dir,
        small_bodies,
        pointings_filtered,
        observatory,
        time_range=time_range,
        tag=tag,
        overwrite=overwrite,
        randomization=randomization,
        output_columns=output_columns,
        chunk_size=chunk_size,
        max_processes=max_processes,
        cleanup=cleanup,
    )

    noise_files: dict[str, str] = {}
    if noise_densities is not None:

        test_data_summary = TestDataSummary.empty()

        # Now generate noise observations
        for noise_density in noise_densities:
            tag_noise = f"{tag}_noise_{noise_density:.3f}"

            # Add noise to the observations
            noise_catalog_file = generate_noise(
                output_dir,
                pointings_filtered,
                observatory,
                noise_density,
                tag=tag_noise,
                seed=seed,
                chunk_size=chunk_size,
                max_processes=max_processes,
                cleanup=cleanup,
            )
            noise_files[f"{noise_density:.2f}"] = noise_catalog_file

            test_data_summary_density = TestDataSummary.from_kwargs(
                catalog_id=[tag],
                start_time=Timestamp.from_mjd(
                    pa.array([pc.min(pointings_filtered.observationStartMJD_TAI)]),
                    scale="tai",
                ),
                end_time=Timestamp.from_mjd(
                    pa.array([pc.max(pointings_filtered.observationStartMJD_TAI)]),
                    scale="tai",
                ),
                num_orbits=[len(small_bodies.orbits)],
                population_name=[small_bodies.name],
                pointings_name=[pointings.name],
                observatory_code=[observatory.code],
                noise_density=[noise_density],
                catalog_file=[catalog_file],
                noise_file=[noise_catalog_file],
            )

            test_data_summary = qv.concatenate(
                [test_data_summary, test_data_summary_density]
            )
            if test_data_summary.fragmented():
                test_data_summary = qv.defragment(test_data_summary)

    else:

        test_data_summary = TestDataSummary.from_kwargs(
            catalog_id=[tag],
            start_time=Timestamp.from_mjd(
                pa.array([pc.min(pointings_filtered.observationStartMJD_TAI)]),
                scale="tai",
            ),
            end_time=Timestamp.from_mjd(
                pa.array([pc.max(pointings_filtered.observationStartMJD_TAI)]),
                scale="tai",
            ),
            num_orbits=[len(small_bodies.orbits)],
            population_name=[small_bodies.name],
            pointings_name=[pointings.name],
            observatory_code=[observatory.code],
            noise_density=None,
            catalog_file=[catalog_file],
            noise_file=None,
        )

    test_data_summary.to_parquet(os.path.join(output_dir, f"{tag}_summary.parquet"))

    return catalog_file, noise_files, test_data_summary
