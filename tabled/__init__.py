"""
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

"""

from tabled.html import get_tables_from_url
from tabled.base import (
    get_table,  # get a table from a variety of sources
    DfFiles,  # a store (mapping from keys to dataframes)
    DfLocalFileReader,  # will be deprecated
    KeyFuncReader,
    DataframeKvReader,
    dflt_ext_mapping,
    df_from_data_according_to_key,
    file_extension,
    # get_file_ext,
    identity,
    key_func_mapping,
    split_keys,
)
from tabled.multi import ColumnOrientedMapping
from tabled.diagnose import (
    dataframe_info,  # extract configurable information from DataFrames
    diagnose_table_collection,  # diagnose a collection of tables
    register_info_func,  # register custom analysis functions
    list_info_funcs,  # list available analysis functions
    scalar_columns,  # get columns that are scalar (CSV-serializable)
)
from tabled.util import (
    upsert_data,  # upsert data from one dataframe to another (add columns/rows as needed)
    duplicate_groups,  # get the groups of duplicates in a dataframe
    collapse_rows,  # collapse rows in a dataframe
    expand_rows,  # expand rows in a dataframe
    collapse_columns,  # collapse columns in a dataframe
    expand_columns,  # expand columns in a dataframe
    auto_decode_bytes,  # Decode a byte sequence into a string, trying charset_normalizer gueses if fails.
    PandasJSONEncoder,  # a json encoder that can handle pandas and numpy objects
    pandas_json_dumps,  # dump a pandas object to a json string
    ensure_columns,  # ensure that the columns are in the dataframe
    ensure_first_columns,  # ensure that the columns are the first columns in the dataframe
    ensure_last_columns,  # ensure that the columns are the last columns in the dataframe
)
from tabled.diagnose import (
    print_dataframe_info,  # print dataframe information
    dataframe_info,  # extract dataframe information as a dictionary
    diagnose_table_collection,  # diagnose a collection of tables
    DFLT_INFO_FUNCS,  # default info functions for dataframe analysis
    scalar_columns,  # get list of columns that are scalar (therefore serializable to CSV)
)
from tabled.wrappers import (
    extension_to_encoder,  # get an encoder for a given extension
    extension_to_decoder,  # get a decoder for a given extension
    extension_based_decoding,  # decode a value based on the extension of the key
    extension_based_encoding,  # encode a value based on the extension of the key
    extension_based_wrap,  # Add extension-based encoding and decoding to a store
    add_extension_codec,  # Add an extension codec
)
from tabled.compare_tables import (
    dataframe_diffs,  # compare two dataframes using specified comparison functions
)

# Import the codecs to register them
import tabled._codecs
