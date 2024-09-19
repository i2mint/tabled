"""
A  (key-value) data-object-layer to get (pandas) tables from a variety of sources with ease.

"""

from tabled.html import get_tables_from_url
from tabled.base import (
    DfFiles,
    DfLocalFileReader,  # will be deprecated
    KeyFuncReader,
    DataframeKvReader,
    dflt_ext_mapping,
    df_from_data_according_to_key,
    get_file_ext,
    identity,
    key_func_mapping,
    split_keys,
)
from tabled.multi import ColumnOrientedMapping
from tabled.util import collapse_rows, expand_rows, collapse_columns, expand_columns
