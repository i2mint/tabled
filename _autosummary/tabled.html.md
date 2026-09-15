# tabled

A data-object-layer package for accessing pandas DataFrames from various sources.

This package provides a unified interface for reading, writing, and manipulating
tabular data from multiple sources including files, URLs, and custom data stores.
It offers flexible key-value mapping functionality where keys can represent file
paths, URLs, or other identifiers, and values are pandas DataFrames.

Key Features:

- Read tables from URLs, HTML pages, and various file formats
- Store abstraction (DfFiles) for mapping keys to DataFrames
- Extension-based encoding/decoding for different file formats
- Column-oriented data manipulation utilities
- DataFrame comparison and diff functionality
- JSON serialization support for pandas objects
- Row/column expansion and collapse operations
- Duplicate detection and handling

Main Components:

- HTML table extraction from web pages
- File-based DataFrame storage with automatic format detection
- Multi-source data readers with customizable key functions
- Utility functions for DataFrame manipulation and analysis
- Codec system for handling different data formats
- Comparison tools for analyzing differences between tables

The package is designed to simplify data pipeline workflows where tabular data
needs to be accessed from heterogeneous sources and processed in a consistent manner.

### Modules

| [`base`](tabled.base.html.md#module-tabled.base)                     | Based functionality for tabled                                                   |
|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`coerce`](tabled.coerce.html.md#module-tabled.coerce)                 | Column type coercion utilities for pandas DataFrames.                            |
| [`compare_tables`](tabled.compare_tables.html.md#module-tabled.compare_tables) | Tools to compare tables                                                          |
| [`diagnose`](tabled.diagnose.html.md#module-tabled.diagnose)             | DataFrame and table collection diagnosis utilities.                              |
| [`html`](tabled.html.html.md#module-tabled.html)                     | To work with html                                                                |
| [`join_tables`](tabled.join_tables.html.md#module-tabled.join_tables)       | Join multiple tables (pandas DataFrames) down to a target subset of columns.     |
| [`misc`](tabled.misc.html.md#module-tabled.misc)                     | Miscellaneous utility functions for tables.                                      |
| [`multi`](tabled.multi.html.md#module-tabled.multi)                   | Multi-tabled data structures.                                                    |
| [`sqlite_tools`](tabled.sqlite_tools.html.md#module-tabled.sqlite_tools)     | General-purpose SQLite to DataFrame/Parquet export tools using DuckDB.           |
| [`tools`](tabled.tools.html.md#module-tabled.tools)                   | Various high-level tools using tabled                                            |
| [`util`](tabled.util.html.md#module-tabled.util)                     | General-purpose utilities for working with DataFrames, dicts, and byte decoding. |
| [`wrappers`](tabled.wrappers.html.md#module-tabled.wrappers)             | Wrapping tools                                                                   |
