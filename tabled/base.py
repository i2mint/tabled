"""Based functionality for tabled"""

# TODO: Completely change so that
#   * routing is more general (not only conditioned on ext, but function of key,
#   and possibly data
#   * Allow on-the-fly parametrization (e.g. sep='\t', index_col=False...)
from functools import partial
from io import BytesIO, StringIO
from typing import (
    Mapping,
    Callable,
    TypeVar,
    KT,
    VT,
    TypeVar,
    Dict,
    Tuple,
    Iterable,
    Union,
)
import os
import re
import pickle

import pandas as pd
from i2 import mk_sentinel, Sig

from dol import Files  # previously: py2store.stores.local_store import LocalBinaryStore
from tabled.util import identity, split_keys


Obj = TypeVar('Obj')
KeyFunc = Callable[[Obj], KT]
Extension = TypeVar('Extension')
DfDecoder = Callable[[Obj], pd.DataFrame]
dflt_not_found_sentinel = mk_sentinel('dflt_not_found_sentinel')


def key_func_mapping(
    obj: Obj,
    mapping: Mapping[KT, VT],
    key: KeyFunc = identity,
    not_found_sentinel=dflt_not_found_sentinel,
) -> VT:
    """Map an object to a value based on a key function"""
    return mapping.get(key(obj), not_found_sentinel)


from dol import KvReader


class KeyFuncReader(KvReader):
    def __init__(self, mapping: Mapping[KT, VT], key: KeyFunc = identity):
        self.mapping = mapping
        self.key = key

    def __getitem__(self, k):
        return self.mapping[self.key(k)]

    def __contains__(self, k):
        return self.key(k) in self.mapping

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __repr__(self):
        return f'{type(self).__name__}({self.mapping}, key={self.key})'


# TODO: Merge with imbed.base extension-based codec mapping, and move here
# TODO: Add some registry functionality to this?
# TODO: Merge with dol's extension-based codec mapping (routing)
dflt_ext_mapping = split_keys(
    {
        'xls xlsx xlsm': partial(pd.read_excel, index=False),  # Excel files
        'csv txt': partial(pd.read_csv, index_col=False),  # CSV and text files
        'tsv': partial(pd.read_csv, sep='\t', index_col=False),  # Tab-separated
        'parquet': pd.read_parquet,  # Parquet format
        'json': partial(pd.read_json, orient='records'),  # JSON format
        'html': partial(pd.read_html, index_col=False),  # HTML tables
        'p pickle pkl': pickle.load,  # Pickle files
        'xml': pd.read_xml,  # XML files
        'h5 hdf5': pd.read_hdf,  # HDF5 format
        'sql sqlite': pd.read_sql,  # SQL queries
        'feather': pd.read_feather,  # Feather format
        'stata dta': pd.read_stata,  # Stata files
        'sas': pd.read_sas,  # SAS files
        # 'gbq': pandas_gbq.read_gbq,  # Google BigQuery
    }
)


def resolve_to_dataframe(
    data, ext: Extension, ext_mapping: Mapping = dflt_ext_mapping, **extra_decoder_kwargs
) -> pd.DataFrame:
    """Get a dataframe from a (data, ext) pair"""
    if ext.startswith('.'):
        ext = ext[1:]
    decoder = key_func_mapping(
        ext,
        ext_mapping,
        key=identity,
        not_found_sentinel=None,  # TODO
    )
    if decoder is not None:
        # pluck out any key-value pairs of extra_decoder_kwargs whose names are
        # arguments of decoder. (This is needed since extra_decoder_kwargs can be
        # servicing multiple decoders, and we don't want to pass arguments to the
        # wrong decoder.)
        extra_decoder_kwargs = Sig(decoder).map_arguments(
            (), extra_decoder_kwargs, allow_partial=True, allow_excess=True
        )
        # decode the data
        return decoder(data, **extra_decoder_kwargs)
    else:
        raise ValueError(f"Don't know how to handle extension: {ext}")


df_from_data_given_ext = resolve_to_dataframe  # back-compatibility alias


def df_from_data_according_to_key(
    data: Obj, mapping: Dict[Extension, DfDecoder], key: KT, **extra_decoder_kwargs
):
    """Get a dataframe from a (data, mapping, key) triple"""
    decoder = key_func_mapping(data, mapping, key=key, not_found_sentinel=None)
    if decoder is None:
        raise ValueError(f"Don't know how to handle key: {key}")
    return decoder(data, **extra_decoder_kwargs)


def get_file_ext(key: KT) -> Extension:
    _, ext = os.path.splitext(key)
    if ext:
        return ext[1:].lower()
    else:
        return ext


protocol_re = re.compile(r'([a-zA-Z0-9]+)://')


def get_protocol(url: str):
    """Get the protocol of a url

    >>> get_protocol('https://www.google.com')
    'https'
    >>> get_protocol('file:///home/user/file.txt')
    'file'

    The function returns None if no protocol is found:

    >>> assert get_protocol('no_protocol_here') is None

    """
    m = protocol_re.match(url)
    if m:
        return m.group(1)


# TODO: Implemented a plugin system (routing) for io resolution concern in disucssion #8
#   See https://github.com/i2mint/tabled/discussions/8#discussion-7519188
#   Just don't know if it's worth the complexity yet.
#   If you find yourself editing default_io_resolver or specifying a lot of custom ones,
#   it might be time to use the plugin system instead.
from typing import BinaryIO

TableSrc = Union[str, bytes, BinaryIO]
BinaryIOCaster = Callable[[TableSrc], BinaryIO]


# TODO: Routing pattern!
def default_io_resolver(src: TableSrc) -> BinaryIO:
    if isinstance(src, str):
        if os.path.isfile(src):
            return open(src, 'rb')
        elif get_protocol(src) in {'http', 'https'}:
            import requests

            return BytesIO(requests.get(src).content)
        else:
            raise ValueError(f"Can't handle source: {src}")
    elif isinstance(src, bytes):
        return BytesIO(src)
    elif hasattr(src, 'read'):
        return src
    else:
        raise ValueError(f"Can't handle source: {src}")


def get_table(
    table_src: TableSrc = None,
    *,
    ext=None,
    mapping=dflt_ext_mapping,
    resolve_to_io=default_io_resolver,
    **extra_decoder_kwargs,
) -> pd.DataFrame:
    """
    Get a table from a variety of sources.
    """
    # If table_src is None, the user is trying to fix the parameters of the function
    if table_src is None:
        return partial(
            ext=ext, mapping=mapping, get_io_obj=get_io_obj, **extra_decoder_kwargs
        )

    # Get a BinaryIO object from the source
    io_reader = resolve_to_io(table_src)

    return resolve_to_dataframe(
        io_reader, ext=ext, ext_mapping=mapping, **extra_decoder_kwargs
    )


# TODO: Make the logic independent from local files assumption.
# TODO: Better separate Reader, and add DfStore to make a writer.
# TODO: Add filtering functionality in init? (By function, regex, extension?)
class DfFiles(Files):
    """A key-value store providing values as pandas.DataFrames.

    Use Case: You have a bunch of files in a folder, all corresponding to some
    dataframes that were saved in some way. You want to a key-value store whose values
    are the (decoded) dataframes corresponding to the files in the folder.

    Args:
    rootdir: A root directory.
    extension_decoder_mapping: A mapping from file extensions to functions that can
        read the dataframes
    extra_decoder_kwargs: Extra arguments to pass to the decoder functions.

    """

    def __init__(
        self,
        rootdir: str,
        extension_decoder_mapping: Dict[Extension, DfDecoder] = dflt_ext_mapping,
        extra_decoder_kwargs: Union[dict, Iterable] = (),
    ):
        super().__init__(rootdir)

        self.key_to_ext = get_file_ext
        extra_decoder_kwargs = dict(extra_decoder_kwargs)
        self.data_and_ext_to_df = partial(
            df_from_data_given_ext,
            mapping=extension_decoder_mapping,
            **extra_decoder_kwargs,
        )

    def __getitem__(self, k):
        ext = self.key_to_ext(k)
        data = BytesIO(super().__getitem__(k))
        return self.data_and_ext_to_df(data, ext)

    def __setitem__(self, k, v):
        raise NotImplementedError('This is a reader: No write operation allowed')

    def __delitem__(self, k):
        raise NotImplementedError('This is a reader: No delete operation allowed')


DfReader = DfFiles  # alias for back-compatibility: TODO: Issue warning on use
DfLocalFileReader = DfFiles  # back-compatibility: TODO: Issue warning on use


# ------------------------------------------------------------------------------
# Wrapping a DataFrame to provide a key-value interface

import pandas as pd
from collections.abc import Mapping


def validate_fields(df, key_fields, value_columns):
    all_columns = df.columns.tolist()
    all_index_levels = list(df.index.names)

    for field in key_fields:
        if field not in all_columns and field not in all_index_levels:
            raise ValueError(
                f"Field {field} not found in the DataFrame columns or index levels."
            )

    for col in value_columns:
        if col not in all_columns:
            raise ValueError(f"Column {col} not found in the DataFrame.")


# TODO: Does it accommodate the case where I want to define my key_fields to be the
# UNNAMED index of the dataframe, plus a column.
# For example, consider (and add to the test) the case where I have df.set_index('A'),
# but remove the name "A" from the index. Now I can't say key_fields=['A', 'B'] anymore.
class DataframeKvReader(Mapping):
    """
    A class to wrap a DataFrame and provide a Mapping interface where keys are
    combinations of specified columns or index levels, and values are sub-dataframes of specified columns.

    Example usage:
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 1],
    ...     'B': [4, 5, 4],
    ...     'C': [7, 8, 9],
    ...     'D': [10, 11, 12]
    ... })
    >>> df
       A  B  C   D
    0  1  4  7  10
    1  2  5  8  11
    2  1  4  9  12
    >>> kv_reader = DataframeKvReader(df, ['A', 'B'], ['C', 'D'])
    >>> key = (1, 4)
    >>> kv_reader[key].reset_index(drop=True)  # doctest: +NORMALIZE_WHITESPACE
       C   D
    0  7  10
    1  9  12
    >>> list(kv_reader) == [(1, 4), (2, 5)]
    True

    But what if one (or more) of the key fields is an index level?
    The DataframeKvReader can handle that too:

    >>> df = df.set_index(['A'])
    >>> df  # doctest: +NORMALIZE_WHITESPACE
       B  C   D
    A
    1  4  7  10
    2  5  8  11
    1  4  9  12
    >>> kv_reader = DataframeKvReader(df, ['A', 'B'], ['C', 'D'])
    >>> key = (1, 4)
    >>> kv_reader[key].reset_index(drop=True)  # doctest: +NORMALIZE_WHITESPACE
       C   D
    0  7  10
    1  9  12
    >>> list(kv_reader) == [(1, 4), (2, 5)]
    True

    """

    def __init__(self, df, key_fields, value_columns=None):
        """
        Initialize the DataframeKvReader.

        Parameters:
        df (pd.DataFrame): The DataFrame to wrap.
        key_fields (str or list of str): Fields (columns or index levels) to use as keys.
        value_columns (str or list of str): Column(s) to use as values. Default is all columns.

        """
        if value_columns is None:
            value_columns = df.columns.tolist()
        validate_fields(df, key_fields, value_columns)
        self.key_fields = key_fields if isinstance(key_fields, list) else [key_fields]
        self.value_columns = (
            value_columns if isinstance(value_columns, list) else [value_columns]
        )

        # Check if key fields are all index levels
        self.is_index_key = all(field in df.index.names for field in self.key_fields)

        if not self.is_index_key:
            # If key fields are not all index levels, reset index and set new index
            self.df = (
                df.reset_index().set_index(self.key_fields, drop=False).sort_index()
            )
        else:
            # If key fields are all index levels, ensure the DataFrame is sorted by those key fields
            self.df = df.sort_index()

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        try:
            sub_df = self.df.loc[key, self.value_columns]
            if isinstance(sub_df, pd.Series):
                sub_df = sub_df.to_frame().T
            return sub_df
        except KeyError:
            raise KeyError(f"Key {key} not found")

    def __iter__(self):
        keys = self.df.index.drop_duplicates().tolist()
        return iter(keys)

    def __len__(self):
        return self.df.index.nunique()

    def __contains__(self, key):
        return key in self.df.index

    def __repr__(self):
        return f"{type(self).__name__}(df=<{len(self.df)} rows>, key_fields={self.key_fields}, value_columns={self.value_columns})"
