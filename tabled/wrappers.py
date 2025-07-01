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

Obj = TypeVar("Obj")
KeyFunc = Callable[[Obj], KT]
Extension = TypeVar("Extension")
DfDecoder = Callable[[Obj], pd.DataFrame]
dflt_not_found_sentinel = mk_sentinel("dflt_not_found_sentinel")


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


protocol_re = re.compile(r"([a-zA-Z0-9]+)://")


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


def save_df_to_zipped_tsv(df: pd.DataFrame, name: str, sep="\t", index=False, **kwargs):
    """Save a dataframe to a zipped tsv file."""
    name = if_extension_present_remove_it(name, ".zip")
    name = if_extension_present_remove_it(name, ".tsv")
    tsv_filepath = f"{name}.tsv"
    zip_filepath = f"{tsv_filepath}.zip"
    df.to_csv(tsv_filepath, sep=sep, index=index, **kwargs)

    file_or_folder_to_zip_file(tsv_filepath, zip_filepath)


from i2 import LiteralVal
from tabled.util import is_instance_of
from operator import methodcaller


def map_values(
    func: Callable,
    d: dict,
    *,
    except_condition=is_instance_of(LiteralVal),
    except_handler=methodcaller("__call__"),
):
    """Map values of a dictionary, except for those that satisfy a condition.

    The `except_condition` is a function that takes a value and returns a boolean.
    If the condition is True, the value is not mapped.
    Instead, the `except_handler` is called with the value, and the result is used as
    the new value (often, the value is left unchanged).

    The default `except_condition` is `is_instance_of(LiteralVal)`, which is a function
    that returns True if the value is to be taken litterally.
    The default `except_handler` is `methodcaller('__call__')`, which will extract
    the litteral value from the `LiteralVal` object.

    >>> map_values(lambda x: x * 10, {'a': 1, 'b': LiteralVal(2), 'c': 3})
    {'a': 10, 'b': 2, 'c': 30}

    """
    return {
        k: except_handler(v) if except_condition(v) else func(v) for k, v in d.items()
    }


# --------------------------------------------------------------------------------------
# The codec mappings

# TODO: Idea: set up the extension_codec plugin system, and use it to set up the default
#   codec mappings, keeping encoder and decoder close.
#   Then, we can have a INCLUDE_INDEX_BY_DEFAULT flat that can help us manage the
#   fact that:
#       if index=True in encoder,  decoder needs to include index_col=0
#       if index=False in encoder, decoder needs to include index_col=None

USE_INDEX = True  # set here for the encoders
INDEX_COL = 0 if USE_INDEX else None  # ... and this will be used for the decoders

import io
import pandas as pd


def single_column_parquet_encode(sequences, col=0):
    """
    Encode a list of sequences into a single-column parquet file.

    >>> sequences_1 = [[1, 2], [3, 4, 5]]
    >>> encoded_1 = single_column_parquet_encode(sequences_1)
    >>> decoded_1 = single_column_parquet_decode(encoded_1)
    >>> all((x == y).all() for x, y in zip(decoded_1, sequences_1))
    True

    """
    return pd.DataFrame({col: sequences}).to_parquet()


def single_column_parquet_decode(b: bytes, col=0):
    """
    Decode a single-column parquet file into a list of sequences.

    >>> sequences_2 = [['one', 'two'], ['three', 'four', 'five']]
    >>> encoded_2 = single_column_parquet_encode(sequences_2)
    >>> decoded_2 = single_column_parquet_decode(encoded_2)
    >>> all((x == y).all() for x, y in zip(decoded_2, sequences_2))
    True

    """
    return pd.read_parquet(io.BytesIO(b))[col].values


_extension_to_encoder = split_keys(
    {
        # csv files
        # note: if index=True, decoder needs to include index_col=0
        #       if index=False, decoder needs to include index_col=None
        "csv txt": partial(pd.DataFrame.to_csv, index=USE_INDEX),
        # tab-separated files
        "tsv": partial(
            pd.DataFrame.to_csv,
            index=USE_INDEX,
            sep="\t",
            escapechar="\\",
            quotechar='"',
        ),
        # json files
        "json": pd.DataFrame.to_json,
        # html tables
        "html": partial(pd.DataFrame.to_html, index=USE_INDEX),
        # pickle files
        "p pickle pkl": pd.DataFrame.to_pickle,
        # numpy arrays
        "npy": LiteralVal(written_bytes(np.save, obj_arg_position_in_writer=1)),
        # zip-compressed tsv (custom implementation)
        "zip": LiteralVal(save_df_to_zipped_tsv),
        # feather format
        "h5 hdf5": pd.DataFrame.to_hdf,
        # stata files
        "stata dta": partial(pd.DataFrame.to_stata, write_index=USE_INDEX),
        # sql queries
        "sql sqlite": pd.DataFrame.to_sql,
        # Google BigQuery
        "gbq": pd.DataFrame.to_gbq,
        # ------------ extensions requiring extra dependencies ------------
        # excel files
        "xls xlsx": partial(
            pd.DataFrame.to_excel, index=USE_INDEX
        ),  # Need: pip install openpyxl, xlrd
        # xml files
        "xml": pd.DataFrame.to_xml,  # Need: pip install lxml
        # parquet format
        "parquet": pd.DataFrame.to_parquet,  # Need: pip install pyarrow, fastparquet
        # feather format
        "single_column_parquet": single_column_parquet_encode,
        "feather": pd.DataFrame.to_feather,  # Need: pip install pyarrow
        # orc format
        "orc": pd.DataFrame.to_orc,  # Need: pip install pyarrow
        # # sas files
        # 'sas': written_bytes(pd.DataFrame.to_sas),  # Need: pip install sas7bdat
        # # SPSS files
        # 'sav': written_bytes(pd.DataFrame.to_spss),  # Need: pip install pyreadstat
    }
)

extension_to_encoder = map_values(written_bytes, _extension_to_encoder)


from dol.util import read_from_bytes

_extension_to_decoder = split_keys(
    {
        # csv and text
        "csv txt": partial(pd.read_csv, index_col=INDEX_COL),
        # tab-separated files
        "tsv": partial(pd.read_csv, sep="\t", index_col=INDEX_COL),
        # parquet format
        "parquet": pd.read_parquet,
        "single_column_parquet": single_column_parquet_decode,
        # json format
        "json": partial(pd.read_json, orient="records"),
        # html tables
        "html": partial(pd.read_html, index_col=INDEX_COL),
        # pickle files
        "p pickle pkl": pickle.load,
        # xml files
        "xml": pd.read_xml,
        # sql queries
        "sql sqlite": pd.read_sql,
        # feather format
        "feather": pd.read_feather,
        # stata files
        "stata dta": partial(pd.read_stata, index_col=INDEX_COL),
        # sas files
        "sas": pd.read_sas,
        # extensions requiring extra dependencies
        # hdf5 format (Hierarchical Data Format)
        "h5 hdf5": pd.read_hdf,  # Need: pip install tables
        # excel files
        "xls xlsx": partial(
            pd.read_excel, index_col=INDEX_COL
        ),  # Need: pip install openpyxl, xlrd
        # parquet format
        "parquet": pd.read_parquet,  # Need: pip install pyarrow, fastparquet
        # feather format
        "feather": pd.read_feather,  # Need: pip install pyarrow
        # orc format
        "orc": pd.read_orc,  # Need: pip install pyarrow
        # SPSS files
        "sav": pd.read_spss,  # Need: pip install pyreadstat
    }
)

extension_to_decoder = map_values(read_from_bytes, _extension_to_decoder)

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


def add_extension_codec(extension=None, *, encoder=None, decoder=None, overwrite=False):
    """
    Add an extension-based encoder and decoder to the extension-code mapping.

    Sure, you could just edit the underlying dictionaries directly, but the design gods
    would not be pleased.

    If no arguments are passed, it will print the current mappings.

    Returns: None (it just adds the in-memory mappings)

    Parameters:
    ----------
    extension: str
        The file extension to add the codec for. If None, it will print the current mappings.
    encoder: callable
        The encoder function to add. If None, it will print the current mappings.
    decoder: callable
        The decoder function to add. If None, it will print the current mappings.
    overwrite: bool
        If True, it will overwrite the existing encoder/decoder for the given extension.
        If False, it will raise a ValueError if the extension already exists.
        Default is False.

    """
    if extension is None and encoder is None and decoder is None:
        # Print the current mappings
        # (The design gods would hate this, for sure)
        return print_current_mappings()

    if encoder is not None:
        if extension in extension_to_encoder and not overwrite:
            raise ValueError(
                f"Encoder for extension '{extension}' already exists. Use 'overwrite=True' to replace it."
            )
        assert callable(encoder), f"Encoder must be a callable. Was: {encoder}"
        extension_to_encoder[extension] = encoder
    if decoder is not None:
        if extension in extension_to_decoder and not overwrite:
            raise ValueError(
                f"Decoder for extension '{extension}' already exists. Use 'overwrite=True' to replace it."
            )
        assert callable(decoder), f"Decoder must be a callable. Was: {decoder}"
        extension_to_decoder[extension] = decoder


# --------------------------------------------------------------------------------------
# Main facades to the codec mappings


def extension_based_decoding(k, v, *, extension_to_decoder=extension_to_decoder):
    """Decode a value based on the extension of the key."""
    ext = file_extension(k)
    decoder = extension_to_decoder.get(ext, None)
    if decoder is None:
        suffix_msg = (
            f"This happened with key: {k}\n"
            "Note that you can add your own extension-based decoder to resolve this."
        )
        if ext != "":
            raise ValueError(
                f"Your DECODER doesn't support this extension: {ext}\n{suffix_msg}"
            )
        else:
            raise ValueError(
                f"Your DECODER doesn't support empty extensions.\n{suffix_msg}"
            )
    return decoder(v)


def extension_based_encoding(k, v, *, extension_to_encoder=extension_to_encoder):
    """Encode a value based on the extension of the key."""
    ext = file_extension(k)
    encoder = extension_to_encoder.get(ext, None)
    if encoder is None:
        suffix_msg = (
            f"This happened with key: {k}\n"
            "Note that you can add your own extension-based encoder to resolve this."
        )
        if ext != "":
            raise ValueError(
                f"Your ENCODER doesn't support this extension: {ext}\n{suffix_msg}"
            )
        else:
            raise ValueError(
                f"Your ENCODER doesn't support empty extensions.\n{suffix_msg}"
            )
    return encoder(v)


@store_decorator
def extension_based_wrap(
    store=None,
    *,
    extension_to_decoder=extension_to_decoder,
    extension_to_encoder=extension_to_encoder,
):
    """Add extension-based encoding and decoding to a store."""
    return wrap_kvs(
        store,
        postget=partial(
            extension_based_decoding, extension_to_decoder=extension_to_decoder
        ),
        preset=partial(
            extension_based_encoding, extension_to_encoder=extension_to_encoder
        ),
    )


extension_based_wrap.dflt_extension_to_decoder = extension_to_decoder
extension_based_wrap.dflt_extension_to_encoder = extension_to_encoder

# ----------------------------------------
# moved from tabled.base


def resolve_to_dataframe(
    data,
    ext: Extension,
    ext_mapping: Mapping = extension_to_decoder,
    **extra_decoder_kwargs,
) -> pd.DataFrame:
    """Get a dataframe from a (data, ext) pair"""
    if ext.startswith("."):
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
            return open(src, "rb")
        elif get_protocol(src) in {"http", "https"}:
            import requests

            return BytesIO(requests.get(src).content)
        elif get_protocol(src) == "graze":
            from graze import graze  # pip install graze

            _, url = src.split("://", 1)
            return BytesIO(graze(url))
        else:
            raise ValueError(f"Can't handle source: {src}")
    elif isinstance(src, bytes):
        return BytesIO(src)
    elif hasattr(src, "read"):
        return src
    else:
        raise ValueError(f"Can't handle source: {src}")
