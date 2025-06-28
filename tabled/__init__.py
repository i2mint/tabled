"""
A  (key-value) data-object-layer to get (pandas) tables from a variety of sources with ease.

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
from tabled.util import (
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
from tabled.misc import (
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
