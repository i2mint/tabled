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
        ext, mapping, key=identity, not_found_sentinel=None,  # TODO
    )
    if trans_func is not None:
        return trans_func(data, **kwargs)
    else:
        raise ValueError(f"Don't know how to handle extension: {ext}")


def df_from_data_according_to_key(data, mapping, key, **kwargs):
    """Get a dataframe from a (data, ext) pair"""
    trans_func = key_func_mapping(data, mapping, key=key, not_found_sentinel=None)
    if trans_func is not None:
        return trans_func(data, **kwargs)
    else:
        raise ValueError(f"Don't know how to handle extension: {ext}")


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
    df_from_data_according_to_key, mapping=dflt_ext_mapping, key=get_ext,
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


class DfLocalFileReader(Files):
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


DfReader = DfLocalFileReader  # alias for back-compatibility: TODO: Issue warning on use
