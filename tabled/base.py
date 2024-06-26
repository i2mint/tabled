"""Based functionality for tabled"""

# TODO: Completely change so that
#   * routing is more general (not only conditioned on ext, but function of key,
#   and possibly data
#   * Allow on-the-fly parametrization (e.g. sep='\t', index_col=False...)
from functools import partial
from io import BytesIO
from typing import Mapping, Callable, TypeVar, KT, VT
import os
import re
import pickle

import pandas as pd
from i2 import mk_sentinel

from dol import Files  # previously: py2store.stores.local_store import LocalBinaryStore
from tabled.util import identity, split_keys


Obj = TypeVar('Obj')
KeyFunc = Callable[[Obj], KT]
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


dflt_ext_mapping = split_keys(
    {
        'xls xlsx': partial(pd.read_excel, index=False),
        'csv': partial(pd.read_csv, index_col=False),
        'tsv': partial(pd.read_csv, sep='\t', index_col=False),
        'json': partial(pd.read_json, orient='records'),
        'html': partial(pd.read_html, index_col=False),
        'p pickle': pickle.load,
    }
)


def df_from_data_given_ext(data, ext, mapping=dflt_ext_mapping, **kwargs):
    """Get a dataframe from a (data, ext) pair"""
    if ext.startswith('.'):
        ext = ext[1:]
    trans_func = key_func_mapping(
        ext,
        mapping,
        key=identity,
        not_found_sentinel=None,  # TODO
    )
    if trans_func is not None:
        return trans_func(data, **kwargs)
    else:
        raise ValueError(f"Don't know how to handle extension: {ext}")


def df_from_data_according_to_key(data, mapping, key, **kwargs):
    """Get a dataframe from a (data, mapping, key) triple"""
    trans_func = key_func_mapping(data, mapping, key=key, not_found_sentinel=None)
    return trans_func(data, **kwargs)


def get_ext(x):
    _, ext = os.path.splitext(x)
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


df_from_data_according_to_ext = partial(
    df_from_data_according_to_key,
    mapping=dflt_ext_mapping,
    key=get_ext,
)

# df_from_data_given_ext meant to be equivalent (but more general, using ext_specs) to
# def df_from_data_given_ext(data, ext, ext_specs=None, **kwargs):
#     """Get a dataframe from a (data, ext) pair"""
#     ext_specs = ext_specs or DFLT_EXT_SPECS  # NOTE: Note used yet
#     if ext.startswith("."):
#         ext = ext[1:]
#     if ext in {"xls", "xlsx"}:
#         kwargs = dict({"index": False}, **kwargs)
#         return pd.read_excel(data, **kwargs)
#     elif ext in {"csv"}:
#         kwargs = dict({"index_col": False}, **kwargs)
#         return pd.read_csv(data, **kwargs)
#     elif ext in {"tsv"}:
#         kwargs = dict({"sep": "\t", "index_col": False}, **kwargs)
#         return pd.read_csv(data, **kwargs)
#     elif ext in {"json"}:
#         kwargs = dict({"orient": "records"}, **kwargs)
#         return pd.read_json(data, **kwargs)
#     elif ext in {"html"}:
#         kwargs = dict({"index_col": False}, **kwargs)
#         return pd.read_html(data, **kwargs)[0]
#     elif ext in {"p", "pickle"}:
#         return pickle.load(data, **kwargs)
#     else:
#         raise ValueError(f"Don't know how to handle extension: {ext}")


# TODO: Make the logic independent from local files assumption.
# TODO: Better separate Reader, and add DfStore to make a writer.


class DfFiles(Files):
    """A key-value store providing values as pandas.DataFrames"""

    def __init__(self, path_format, mapping=dflt_ext_mapping):
        super().__init__(path_format)

        self.key_to_ext = get_ext
        self.data_and_ext_to_df = partial(df_from_data_given_ext, mapping=mapping)

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


# Test cases
def test_DataframeKvReader():
    df = pd.DataFrame(
        {'A': [1, 2, 1], 'B': [4, 5, 4], 'C': [7, 8, 9], 'D': [10, 11, 12]}
    )

    kv_readers = [
        DataframeKvReader(df, ['A', 'B'], ['C', 'D']),
        DataframeKvReader(df.set_index(['A', 'B']), ['A', 'B'], ['C', 'D']),
        DataframeKvReader(df.set_index('A'), ['A', 'B'], ['C', 'D']),
    ]

    for kv_reader in kv_readers:
        # Accessing values
        key = (1, 4)
        expected_df = pd.DataFrame([{'C': 7, 'D': 10}, {'C': 9, 'D': 12}], index=[0, 2])
        assert (
            kv_reader[key]
            .reset_index(drop=True)
            .equals(expected_df.reset_index(drop=True))
        )
        assert list(kv_reader) == [(1, 4), (2, 5)]
