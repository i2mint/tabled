"""Wrapping tools

A lot of what is defined here are functions that are used to transform data.
More precisely, encode and decode data depending on it's format, file extension, etc.

"""

import os
import re
import pickle
from functools import partial
import io
from io import BytesIO
import pandas as pd
import json
import pickle
from posixpath import splitext
from typing import Mapping, Union, Callable, Dict, TypeVar, KT, VT, BinaryIO

import numpy as np
from dol import Pipe, wrap_kvs, written_bytes, store_decorator
from dol.zipfiledol import file_or_folder_to_zip_file
from i2 import mk_sentinel, Sig

from tabled.util import identity, split_keys, auto_decode_bytes

# TODO: Merge with imbed.base extension-based codec mapping, and move here
# TODO: Add some registry functionality to this?
# TODO: Merge with dol's extension-based codec mapping (routing)
# TODO: Merge with codec-matching ("routing"?) functionalities of dol
# TODO: Move the extension-based codec stuff to tabled, replacing current DfFiles etc.

Obj = TypeVar('Obj')
KeyFunc = Callable[[Obj], KT]
Extension = TypeVar('Extension')
DfDecoder = Callable[[Obj], pd.DataFrame]
dflt_not_found_sentinel = mk_sentinel('dflt_not_found_sentinel')


# --------------------------------------------------------------------------------------
# Helpers for file extensions and protocols


def _get_extension(path: str) -> str:
    return splitext(path)[1]


def _lower_case_without_dot_prefix(ext: str) -> str:
    return ext[1:].lower()


def _key_property(
    key: KT, *, extract_prop=_get_extension, prop_egress=_lower_case_without_dot_prefix
):
    ext = extract_prop(key)
    return prop_egress(ext)


def file_extension(key: str) -> Extension:
    """Get the file extension from a key

    >>> file_extension('hello.world')
    'world'
    >>> file_extension('hello')
    ''
    """
    return _key_property(
        key, extract_prop=_get_extension, prop_egress=_lower_case_without_dot_prefix
    )


get_file_ext = file_extension  # back-compatibility alias


# TODO: Deprecated
def get_extension(key: str) -> str:
    """Return the extension of a file path.

    Note that it includes the dot.

    >>> get_extension('hello.world')  # doctest: +SKIP
    '.world'

    If there's no extension, it returns an empty string.

    >>> get_extension('hello')  # doctest: +SKIP
    ''

    """
    print(f"Deprecating get_extension. Consider using ")
    return _key_property(key, extract_prop=_get_extension, prop_egress=identity)


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


def key_func_mapping(
    obj: Obj,
    mapping: Mapping[KT, VT],
    key: KeyFunc = identity,
    not_found_sentinel=dflt_not_found_sentinel,
) -> VT:
    """Map an object to a value based on a key function"""
    return mapping.get(key(obj), not_found_sentinel)


# --------------------------------------------------------------------------------------
# Helpers for the codec mappings


def if_extension_not_present_add_it(filepath, extension):
    if not filepath.endswith(extension):
        return filepath + extension
    return filepath


def if_extension_present_remove_it(filepath, extension):
    if filepath.endswith(extension):
        return filepath[: -len(extension)]
    return filepath


def save_df_to_zipped_tsv(df: pd.DataFrame, name: str, sep='\t', index=False, **kwargs):
    """Save a dataframe to a zipped tsv file."""
    name = if_extension_present_remove_it(name, '.zip')
    name = if_extension_present_remove_it(name, '.tsv')
    tsv_filepath = f'{name}.tsv'
    zip_filepath = f'{tsv_filepath}.zip'
    df.to_csv(tsv_filepath, sep=sep, index=index, **kwargs)

    file_or_folder_to_zip_file(tsv_filepath, zip_filepath)


# --------------------------------------------------------------------------------------
# The codec mappings

extension_to_encoder = {
    'html': written_bytes(pd.DataFrame.to_html),
    'json': written_bytes(pd.DataFrame.to_json),
    'pkl': written_bytes(pd.DataFrame.to_pickle),
    'parquet': written_bytes(pd.DataFrame.to_parquet, obj_arg_position_in_writer=0),
    'npy': written_bytes(np.save, obj_arg_position_in_writer=1),
    'csv': written_bytes(pd.DataFrame.to_csv),
    'xlsx': written_bytes(pd.DataFrame.to_excel),
    'tsv': written_bytes(
        partial(pd.DataFrame.to_csv, sep='\t', escapechar='\\', quotechar='"')
    ),
    # complete with all the other pd.DataFrame.to_... methods
    'zip': save_df_to_zipped_tsv,
    'feather': written_bytes(pd.DataFrame.to_feather),
    'h5': written_bytes(pd.DataFrame.to_hdf),
    'dta': written_bytes(pd.DataFrame.to_stata),
    'sql': written_bytes(pd.DataFrame.to_sql),
    'gbq': written_bytes(pd.DataFrame.to_gbq),
}

from dol.util import read_from_bytes

extension_to_decoder = split_keys(
    {
        'xls xlsx xlsm': partial(pd.read_excel, index=False),  # Excel files
        'csv txt': partial(pd.read_csv, index_col=False),  # CSV and text files
        'tsv': partial(pd.read_csv, sep='\t', index_col=False),  # Tab-separated
        'parquet': read_from_bytes(pd.read_parquet),  # Parquet format
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

dflt_ext_mapping = extension_to_decoder  # back-compatibility alias


# --------------------------------------------------------------------------------------
# Operations on the codec mappings


def get_codec_mappings(
    *,
    extension_to_encoder=extension_to_encoder,
    extension_to_decoder=extension_to_decoder,
):
    return dict(
        encoders=extension_to_encoder,
        decoders=extension_to_decoder,
    )


def print_current_mappings():
    from pprint import pprint

    pprint("Current encoder and decoder mappings:")
    pprint(get_codec_mappings())


def add_extension_codec(extension=None, *, encoder=None, decoder=None):
    """
    Add an extension-based encoder and decoder to the extension-code mapping.

    Sure, you could just edit the underlying dictionaries directly, but the design gods
    would not be pleased.

    If no arguments are passed, it will print the current mappings.

    Returns: None (it just adds the in-memory mappings)

    """
    if extension is None and encoder is None and decoder is None:
        # Print the current mappings
        # (The design gods would hate this, for sure)
        return print_current_mappings()

    if encoder is not None:
        extension_to_encoder[extension] = encoder
    if decoder is not None:
        extension_to_decoder[extension] = decoder


# --------------------------------------------------------------------------------------
# Main facades to the codec mappings


def extension_based_decoding(k, v, *, extension_to_decoder=extension_to_decoder):
    """Decode a value based on the extension of the key."""
    ext = file_extension(k)
    decoder = extension_to_decoder.get(ext, None)
    if decoder is None:
        raise ValueError(f"Unknown extension: {ext}")
    return decoder(v)


def extension_based_encoding(k, v, *, extension_to_encoder=extension_to_encoder):
    """Encode a value based on the extension of the key."""
    ext = file_extension(k)
    encoder = extension_to_encoder.get(ext, None)
    if encoder is None:
        raise ValueError(f"Unknown extension: {ext}")
    return encoder(v)


@store_decorator
def extension_base_wrap(store=None):
    """Add extension-based encoding and decoding to a store."""
    return wrap_kvs(
        store,
        postget=extension_based_decoding,
        preset=extension_based_encoding,
    )


# ----------------------------------------
# moved from tabled.base


def resolve_to_dataframe(
    data,
    ext: Extension,
    ext_mapping: Mapping = extension_to_decoder,
    **extra_decoder_kwargs,
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


# TODO: Implemented a plugin system (routing) for io resolution concern in disucssion #8
#   See https://github.com/i2mint/tabled/discussions/8#discussion-7519188
#   Just don't know if it's worth the complexity yet.
#   If you find yourself editing default_io_resolver or specifying a lot of custom ones,
#   it might be time to use the plugin system instead.

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
        elif get_protocol(src) == 'graze':
            from graze import graze  # pip install graze

            _, url = src.split("://", 1)
            return BytesIO(graze(url))
        else:
            raise ValueError(f"Can't handle source: {src}")
    elif isinstance(src, bytes):
        return BytesIO(src)
    elif hasattr(src, 'read'):
        return src
    else:
        raise ValueError(f"Can't handle source: {src}")
