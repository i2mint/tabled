# tabled.sqlite_tools

General-purpose SQLite to DataFrame/Parquet export tools using DuckDB.

This module provides utilities for extracting data from SQLite databases and exporting
it to pandas DataFrames or Parquet files. It uses DuckDB with the sqlite_scanner
extension for efficient data extraction.

Key functions:

- export_sqlite_to_dataframes: Extract SQLite tables to pandas DataFrames
- export_sqlite_to_parquet: Export SQLite tables directly to Parquet files
- export_sqlite_to_dataframes_and_parquet: Combined export to both formats

All functions use DuckDB’s sqlite_scanner extension which provides fast, efficient
access to SQLite databases without loading the entire database into memory.

### Functions

| [`export_sqlite_query_to_parquet`](#tabled.sqlite_tools.export_sqlite_query_to_parquet)(...[, ...])      | Export an arbitrary SQL query (against the attached SQLite DB) to a Parquet file.   |
|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`export_sqlite_to_dataframes`](#tabled.sqlite_tools.export_sqlite_to_dataframes)(sqlite_db_file, \*) | Export tables from SQLite to pandas DataFrames using DuckDB + sqlite_scanner.       |
| [`export_sqlite_to_dataframes_and_parquet`](#tabled.sqlite_tools.export_sqlite_to_dataframes_and_parquet)(...)    | Export SQLite tables to both DataFrames and Parquet files.                          |
| [`export_sqlite_to_parquet`](#tabled.sqlite_tools.export_sqlite_to_parquet)(sqlite_db_file, ...)   | Export tables from a SQLite .db file to Parquet using DuckDB + sqlite_scanner.      |

### tabled.sqlite_tools.export_sqlite_query_to_parquet(sqlite_db_file, out_path, , query, schema='src', compression='ZSTD', install_extensions=True, verbose=False)

Export an arbitrary SQL query (against the attached SQLite DB) to a Parquet file.

Useful for generating:

- edge lists (source/target)
- node tables (id + attributes)
- filtered subsets

### Example

```python
export_sqlite_query_to_parquet(
    "my.db",
    "edges.parquet",
    query="SELECT from_id AS source, to_id AS target, weight FROM edges",
)
```

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### tabled.sqlite_tools.export_sqlite_to_dataframes(sqlite_db_file, , tables=None, schema='src', install_extensions=True, verbose=False)

Export tables from SQLite to pandas DataFrames using DuckDB + sqlite_scanner.

* **Parameters:**
  * **sqlite_db_file** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the SQLite database file (.db / .sqlite / .sqlite3).
  * **tables** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Optional list of table names to export. If None, exports all discovered tables.
  * **schema** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – DuckDB schema name to attach the SQLite DB as (default “src”).
  * **install_extensions** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, runs INSTALL/LOAD sqlite_scanner (helpful for first run).
  * **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Print progress.
* **Returns:**
  Dictionary mapping table names to DataFrames.
* **Return type:**
  [`Dict`](https://docs.python.org/3/library/typing.html#typing.Dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), `DataFrame`]

### tabled.sqlite_tools.export_sqlite_to_dataframes_and_parquet(sqlite_db_file, out_dir=None, , tables=None, schema='src', compression='ZSTD', overwrite=True, install_extensions=True, verbose=False)

Export SQLite tables to both DataFrames and Parquet files.

This is a combined function that exports SQLite tables to pandas DataFrames
and optionally saves them to Parquet files in a single operation.

* **Parameters:**
  * **sqlite_db_file** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the SQLite database file (.db / .sqlite / .sqlite3).
  * **out_dir** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Optional directory where Parquet files will be written.
    If None, only DataFrames are returned.
  * **tables** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Optional list of table names to export. If None, exports all tables.
  * **schema** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – DuckDB schema name to attach the SQLite DB as (default “src”).
  * **compression** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Parquet compression codec. Common: “ZSTD”, “SNAPPY”, “GZIP”, “NONE”.
  * **overwrite** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If False, skip exporting tables where parquet files already exist.
  * **install_extensions** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, runs INSTALL/LOAD sqlite_scanner (helpful for first run).
  * **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Print progress information.
* **Returns:**
  A tuple containing:
  - Dictionary mapping table names to DataFrames
  - Output directory path (if out_dir was provided)
* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`Dict`](https://docs.python.org/3/library/typing.html#typing.Dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), `DataFrame`], [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]]

### tabled.sqlite_tools.export_sqlite_to_parquet(sqlite_db_file, out_dir, , tables=None, schema='src', compression='ZSTD', overwrite=True, install_extensions=True, verbose=False)

Export tables from a SQLite .db file to Parquet using DuckDB + sqlite_scanner.

This is a general-purpose exporter that:

- attaches the SQLite file to DuckDB
- discovers tables (or uses the provided list)
- writes each table to <out_dir>/<table>.parquet

* **Parameters:**
  * **sqlite_db_file** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the SQLite database file (.db / .sqlite / .sqlite3).
  * **out_dir** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Directory where Parquet files will be written.
  * **tables** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Optional list of table names to export. If None, exports all discovered tables.
  * **schema** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – DuckDB schema name to attach the SQLite DB as (default “src”).
  * **compression** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Parquet compression codec. Common: “ZSTD”, “SNAPPY”, “GZIP”, “NONE”.
  * **overwrite** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If False, skip exporting a table if the target parquet file already exists.
  * **install_extensions** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, runs INSTALL/LOAD sqlite_scanner (helpful for first run).
  * **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Print progress.
* **Returns:**
  The output directory (resolved).
* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)
