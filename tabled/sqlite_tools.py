"""General-purpose SQLite to DataFrame/Parquet export tools using DuckDB.

This module provides utilities for extracting data from SQLite databases and exporting
it to pandas DataFrames or Parquet files. It uses DuckDB with the sqlite_scanner
extension for efficient data extraction.

Key functions:
- export_sqlite_to_dataframes: Extract SQLite tables to pandas DataFrames
- export_sqlite_to_parquet: Export SQLite tables directly to Parquet files
- export_sqlite_to_dataframes_and_parquet: Combined export to both formats

All functions use DuckDB's sqlite_scanner extension which provides fast, efficient
access to SQLite databases without loading the entire database into memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

DFLT_VERBOSE = False

# Core / standard SQLite database extensions
SQLITE_EXTENSIONS_CORE = [
    ".db",
    ".sqlite",
    ".sqlite3",
]

# Common alternative extensions used by tools and platforms
SQLITE_EXTENSIONS_COMMON = [
    ".db3",
    ".s3db",
    ".sl3",
    ".sq3",
]

# Application-specific extensions that are often SQLite under the hood
SQLITE_EXTENSIONS_APP_SPECIFIC = [
    ".dat",
    ".store",
    ".cache",
    ".catalog",
    ".wallet",
    ".browser",
]

# SQLite auxiliary / transient files (usually NOT the main database)
SQLITE_EXTENSIONS_AUXILIARY = [
    ".db-journal",
    ".sqlite-journal",
    ".db-wal",
    ".sqlite-wal",
    ".db-shm",
    ".sqlite-shm",
]


def export_sqlite_to_dataframes(
    sqlite_db_file: Union[str, Path],
    *,
    tables: Optional[Sequence[str]] = None,
    schema: str = "src",
    install_extensions: bool = True,
    verbose: bool = DFLT_VERBOSE,
) -> Dict[str, "pd.DataFrame"]:
    """
    Export tables from SQLite to pandas DataFrames using DuckDB + sqlite_scanner.

    Parameters
    ----------
    sqlite_db_file:
        Path to the SQLite database file (.db / .sqlite / .sqlite3).
    tables:
        Optional list of table names to export. If None, exports all discovered tables.
    schema:
        DuckDB schema name to attach the SQLite DB as (default "src").
    install_extensions:
        If True, runs INSTALL/LOAD sqlite_scanner (helpful for first run).
    verbose:
        Print progress.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping table names to DataFrames.
    """
    import duckdb

    # Verbose logging helper
    clog = (lambda *args: print(*args)) if verbose else (lambda *args: None)

    sqlite_db_file = Path(sqlite_db_file)
    if not sqlite_db_file.exists():
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_db_file}")

    clog(f"DuckDB: connecting to extract DataFrames")

    con = duckdb.connect()
    try:
        if install_extensions:
            clog("DuckDB: installing/loading sqlite_scanner extension...")
            con.execute("INSTALL sqlite_scanner;")
            con.execute("LOAD sqlite_scanner;")
        else:
            con.execute("LOAD sqlite_scanner;")

        # Attach the SQLite database
        clog(f"DuckDB: attaching SQLite db as schema '{schema}'...")
        con.execute(f"ATTACH '{sqlite_db_file.as_posix()}' AS {schema} (TYPE sqlite);")

        # Determine tables to export
        if tables is None:
            discovered = con.execute(f"SHOW TABLES FROM {schema};").fetchall()
            tables = [t[0] for t in discovered]
            clog("Tables discovered:", tables)
        else:
            tables = list(tables)
            clog("Tables specified:", tables)

        # Export each table to DataFrame
        dataframes = {}
        for table_name in tables:
            clog(f"Loading {schema}.{table_name} -> DataFrame")
            df = con.execute(f"SELECT * FROM {schema}.{table_name}").df()
            dataframes[table_name] = df

        clog("Done extracting DataFrames.")
        return dataframes

    finally:
        con.close()


def export_sqlite_to_parquet(
    sqlite_db_file: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    tables: Optional[Sequence[str]] = None,
    schema: str = "src",
    compression: str = "ZSTD",
    overwrite: bool = True,
    install_extensions: bool = True,
    verbose: bool = DFLT_VERBOSE,
) -> Path:
    """
    Export tables from a SQLite .db file to Parquet using DuckDB + sqlite_scanner.

    This is a general-purpose exporter:
      - attaches the SQLite file to DuckDB
      - discovers tables (or uses the provided list)
      - writes each table to <out_dir>/<table>.parquet

    Parameters
    ----------
    sqlite_db_file:
        Path to the SQLite database file (.db / .sqlite / .sqlite3).
    out_dir:
        Directory where Parquet files will be written.
    tables:
        Optional list of table names to export. If None, exports all discovered tables.
    schema:
        DuckDB schema name to attach the SQLite DB as (default "src").
    compression:
        Parquet compression codec. Common: "ZSTD", "SNAPPY", "GZIP", "NONE".
    overwrite:
        If False, skip exporting a table if the target parquet file already exists.
    install_extensions:
        If True, runs INSTALL/LOAD sqlite_scanner (helpful for first run).
    verbose:
        Print progress.

    Returns
    -------
    Path
        The output directory (resolved).
    """
    import duckdb

    # Verbose logging helper
    clog = (lambda *args: print(*args)) if verbose else (lambda *args: None)

    sqlite_db_file = Path(sqlite_db_file)
    if not sqlite_db_file.exists():
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_db_file}")

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clog(f"DuckDB: connecting (export -> {out_dir})")

    con = duckdb.connect()
    try:
        if install_extensions:
            clog("DuckDB: installing/loading sqlite_scanner extension...")
            con.execute("INSTALL sqlite_scanner;")
            con.execute("LOAD sqlite_scanner;")
        else:
            con.execute("LOAD sqlite_scanner;")

        # Attach the SQLite database under a schema name (e.g. src)
        clog(f"DuckDB: attaching SQLite db as schema '{schema}'...")
        con.execute(f"ATTACH '{sqlite_db_file.as_posix()}' AS {schema} (TYPE sqlite);")

        # Determine tables to export
        if tables is None:
            # DuckDB returns one column: table name
            discovered = con.execute(f"SHOW TABLES FROM {schema};").fetchall()
            tables = [t[0] for t in discovered]
            clog("Tables discovered:", tables)
        else:
            tables = list(tables)
            clog("Tables specified:", tables)

        # Export each table
        for table_name in tables:
            out_path = out_dir / f"{table_name}.parquet"

            if not overwrite and out_path.exists():
                clog(f"Skipping {table_name} (exists): {out_path}")
                continue

            clog(f"Exporting {schema}.{table_name} -> {out_path}")

            # COPY ... TO writes Parquet directly
            con.execute(
                f"""
                COPY (SELECT * FROM {schema}.{table_name})
                TO '{out_path.as_posix()}'
                (FORMAT PARQUET, COMPRESSION {compression});
                """
            )

        clog("Done.")
        return out_dir

    finally:
        con.close()


def export_sqlite_query_to_parquet(
    sqlite_db_file: Union[str, Path],
    out_path: Union[str, Path],
    *,
    query: str,
    schema: str = "src",
    compression: str = "ZSTD",
    install_extensions: bool = True,
    verbose: bool = DFLT_VERBOSE,
) -> Path:
    """
    Export an arbitrary SQL query (against the attached SQLite DB) to a Parquet file.

    Useful for generating:
      - edge lists (source/target)
      - node tables (id + attributes)
      - filtered subsets

    Example:
        export_sqlite_query_to_parquet(
            "my.db",
            "edges.parquet",
            query="SELECT from_id AS source, to_id AS target, weight FROM edges"
        )
    """
    import duckdb

    # Verbose logging helper
    clog = (lambda *args: print(*args)) if verbose else (lambda *args: None)

    sqlite_db_file = Path(sqlite_db_file)
    if not sqlite_db_file.exists():
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_db_file}")

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        if install_extensions:
            con.execute("INSTALL sqlite_scanner;")
            con.execute("LOAD sqlite_scanner;")
        else:
            con.execute("LOAD sqlite_scanner;")

        con.execute(f"ATTACH '{sqlite_db_file.as_posix()}' AS {schema} (TYPE sqlite);")

        # If caller wrote query against SQLite tables without schema prefix,
        # they can include it themselves. We also allow a convenience placeholder:
        #   {schema} in the query string.
        formatted_query = query.format(schema=schema)

        clog(f"Exporting query -> {out_path}")
        con.execute(
            f"""
            COPY ({formatted_query})
            TO '{out_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION {compression});
            """
        )

        return out_path
    finally:
        con.close()


def export_sqlite_to_dataframes_and_parquet(
    sqlite_db_file: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    *,
    tables: Optional[Sequence[str]] = None,
    schema: str = "src",
    compression: str = "ZSTD",
    overwrite: bool = True,
    install_extensions: bool = True,
    verbose: bool = DFLT_VERBOSE,
) -> tuple[Dict[str, "pd.DataFrame"], Optional[Path]]:
    """Export SQLite tables to both DataFrames and Parquet files.

    This is a combined function that exports SQLite tables to pandas DataFrames
    and optionally saves them to Parquet files in a single operation.

    Parameters
    ----------
    sqlite_db_file:
        Path to the SQLite database file (.db / .sqlite / .sqlite3).
    out_dir:
        Optional directory where Parquet files will be written.
        If None, only DataFrames are returned.
    tables:
        Optional list of table names to export. If None, exports all tables.
    schema:
        DuckDB schema name to attach the SQLite DB as (default "src").
    compression:
        Parquet compression codec. Common: "ZSTD", "SNAPPY", "GZIP", "NONE".
    overwrite:
        If False, skip exporting tables where parquet files already exist.
    install_extensions:
        If True, runs INSTALL/LOAD sqlite_scanner (helpful for first run).
    verbose:
        Print progress information.

    Returns
    -------
    tuple[Dict[str, pd.DataFrame], Optional[Path]]
        A tuple containing:
        - Dictionary mapping table names to DataFrames
        - Output directory path (if out_dir was provided)
    """
    """
    Export tables from SQLite to both DataFrames AND Parquet files.

    More efficient than calling both functions separately since it reuses
    the same DuckDB connection and table discovery.

    Parameters
    ----------
    sqlite_db_file:
        Path to the SQLite database file.
    out_dir:
        Directory where Parquet files will be written. If None, only DataFrames returned.
    tables:
        Optional list of table names to export. If None, exports all discovered tables.
    schema:
        DuckDB schema name to attach the SQLite DB as (default "src").
    compression:
        Parquet compression codec.
    overwrite:
        If False, skip exporting a table if the target parquet file already exists.
    install_extensions:
        If True, runs INSTALL/LOAD sqlite_scanner.
    verbose:
        Print progress.

    Returns
    -------
    tuple[Dict[str, pd.DataFrame], Optional[Path]]
        (dataframes_dict, output_directory_path_or_none)
    """
    import duckdb

    # Verbose logging helper
    clog = (lambda *args: print(*args)) if verbose else (lambda *args: None)

    sqlite_db_file = Path(sqlite_db_file)
    if not sqlite_db_file.exists():
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_db_file}")

    if out_dir is not None:
        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        clog(f"DuckDB: connecting (DataFrames + export -> {out_dir})")
    else:
        clog(f"DuckDB: connecting (DataFrames only)")

    con = duckdb.connect()
    try:
        if install_extensions:
            clog("DuckDB: installing/loading sqlite_scanner extension...")
            con.execute("INSTALL sqlite_scanner;")
            con.execute("LOAD sqlite_scanner;")
        else:
            con.execute("LOAD sqlite_scanner;")

        # Attach the SQLite database
        clog(f"DuckDB: attaching SQLite db as schema '{schema}'...")
        con.execute(f"ATTACH '{sqlite_db_file.as_posix()}' AS {schema} (TYPE sqlite);")

        # Determine tables to export
        if tables is None:
            discovered = con.execute(f"SHOW TABLES FROM {schema};").fetchall()
            tables = [t[0] for t in discovered]
            clog("Tables discovered:", tables)
        else:
            tables = list(tables)
            clog("Tables specified:", tables)

        # Extract DataFrames and optionally save to Parquet
        dataframes = {}
        for table_name in tables:
            clog(f"Loading {schema}.{table_name}")
            df = con.execute(f"SELECT * FROM {schema}.{table_name}").df()
            dataframes[table_name] = df

            # Optionally save to Parquet
            if out_dir is not None:
                out_path = out_dir / f"{table_name}.parquet"

                if not overwrite and out_path.exists():
                    clog(f"Skipping {table_name} (exists): {out_path}")
                    continue

                clog(f"Saving {table_name} -> {out_path}")
                df.to_parquet(out_path, compression=compression.lower())

        clog("Done.")
        return dataframes, out_dir

    finally:
        con.close()
