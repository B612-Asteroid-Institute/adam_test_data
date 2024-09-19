import sqlite3 as sql

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv


class Pointings(qv.Table):

    # Columns required by Sorcha
    observationId = qv.LargeStringColumn()
    observationStartMJD_TAI = qv.Float64Column()
    visitTime = qv.Float64Column()
    visitExposureTime = qv.Float64Column()
    filter = qv.LargeStringColumn()
    seeingFwhmGeom_arcsec = qv.Float64Column()
    seeingFwhmEff_arcsec = qv.Float64Column()
    fieldFiveSigmaDepth_mag = qv.Float64Column()
    fieldRA_deg = qv.Float64Column()
    fieldDec_deg = qv.Float64Column()
    rotSkyPos_deg = qv.Float64Column()

    name = qv.StringAttribute()

    # Additional columns which may be useful
    observatory_code = qv.LargeStringColumn(nullable=True)

    def to_sql(self, con: sql.Connection, table_name: str = "pointings") -> None:
        """
        Save the table to an SQLite database for sorcha to use.

        Parameters
        ----------
        con : sqlite3.Connection
            The connection to the SQLite database.
        """
        # Create an attribute table and store the name attribute
        con.execute("""CREATE TABLE IF NOT EXISTS attributes (name TEXT)""")
        con.execute("""INSERT INTO attributes (name) VALUES (?)""", (self.name,))
        self.to_dataframe().to_sql(table_name, con, if_exists="replace", index=False)

    @classmethod
    def from_sql(
        cls, con: sql.Connection, table_name: str = "pointings"
    ) -> "Pointings":
        """
        Load the table from an SQLite database.

        Parameters
        ----------
        con : sqlite3.Connection
            The connection to the SQLite database.

        Returns
        -------
        Pointings
            The table loaded from the database.
        """
        name = con.execute("""SELECT name FROM attributes""").fetchone()[0]
        query = f"""SELECT * FROM {table_name}"""
        return cls.from_dataframe(pd.read_sql(query, con, index_col=None), name=name)

    def exposure_midpoint(self) -> pa.Array:
        """
        Calculate the midpoint of the exposures.

        Returns
        -------
        pa.Array
            The midpoint of each exposure in MJD TAI.
        """
        return pc.add(
            self.observationStartMJD_TAI,
            pc.divide(self.visitExposureTime, 2.0 / 86400.0),
        )
