"""
A  (key-value) data-object-layer to get (pandas) tables from a variety of sources with ease.

"""
from tabled.html import get_tables_from_url
from tabled.base import (
    DfLocalFileReader,
    KeyFuncReader,
    dflt_ext_mapping,
    df_from_data_according_to_key,
    get_ext,
    identity,
    key_func_mapping,
    split_keys,
)
