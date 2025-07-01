"""Based functionality for tabled"""

# TODO: Completely change so that
#   * routing is more general (not only conditioned on ext, but function of key,
#   and possibly data
#   * Allow on-the-fly parametrization (e.g. sep='\t', index_col=False...)
from functools import partial
from io import BytesIO
from typing import (
    Mapping,
    KT,
    VT,
    Dict,
    Iterable,
    Union,
)

import pandas as pd

from dol import Files, KvReader

from tabled.util import identity, split_keys


from tabled.wrappers import (
    TableSrc,
    KeyFunc,
    Extension,
    DfDecoder,
    key_func_mapping,  # Map an object to a value based on a key function
    dflt_ext_mapping,  # Default extension mapping for various file types (DEPRECATED)
    resolve_to_dataframe,  # Get a dataframe from a (data, ext) pair
    df_from_data_given_ext,  # Alias for resolve_to_dataframe for backward compatibility
    df_from_data_according_to_key,  # Get a dataframe from a (data, mapping, key) triple
    file_extension,  # Get the file extension from a key
    get_protocol,  # Get the protocol of a URL
    default_io_resolver,  # Default IO resolver for various sources
)


# Obj = TypeVar('Obj')
# KeyFunc = Callable[[Obj], KT]
# Extension = TypeVar('Extension')
# DfDecoder = Callable[[Obj], pd.DataFrame]
# dflt_not_found_sentinel = mk_sentinel('dflt_not_found_sentinel')


# def key_func_mapping(
#     obj: Obj,
#     mapping: Mapping[KT, VT],
#     key: KeyFunc = identity,
#     not_found_sentinel=dflt_not_found_sentinel,
# ) -> VT:
#     """Map an object to a value based on a key function"""
#     return mapping.get(key(obj), not_found_sentinel)


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
        return f"{type(self).__name__}({self.mapping}, key={self.key})"


def convert_collection_to_dataframe_if_possible(x):
    if isinstance(x, pd.DataFrame):
        return x
    elif isinstance(x, (dict, list, tuple)):
        return pd.DataFrame(x)
    elif isinstance(x, (pd.Series, pd.Index, pd.MultiIndex)):
        return x.to_frame()
    elif isinstance(x, pd.Index):
        return x.to_frame()
    else:
        return x


# TODO: Handle the zip case more cleanly, and generalize to other double-extensions
def get_table(
    table_src: TableSrc = None,
    *,
    ext=None,
    ext_mapping=dflt_ext_mapping,
    resolve_to_io=default_io_resolver,
    **extra_decoder_kwargs,
) -> pd.DataFrame:
    """
    Get a table from a variety of sources.
    """
    # If table_src is None, the user is trying to fix the parameters of the function

    if table_src is None:
        return partial(
            get_table,
            ext=ext,
            ext_mapping=ext_mapping,
            resolve_to_io=resolve_to_io,
            **extra_decoder_kwargs,
        )

    table_src = convert_collection_to_dataframe_if_possible(table_src)
    if isinstance(table_src, pd.DataFrame):
        return table_src

    if ext is None:
        ext = ""

    if not ext and isinstance(table_src, str):
        key = table_src
        ext = file_extension(key)
        if ext == "zip":
            # compute the next extension
            next_ext = file_extension(key[: -len(".zip")])
            if next_ext:
                ext = f"{next_ext}.{ext}"  # e.g. 'csv.zip'

    # TODO: Here's a great waste, since many of our table reading functions can
    #       take file-like (paths, io objects) as input. Should make wrappers.py so that
    #       one decoder can be given, and a input-type aware function can transform
    #       the input to the right type for that particular decoder
    # convert to bytes if not so already
    if not isinstance(table_src, bytes):
        io_reader = resolve_to_io(table_src)
        table_src = io_reader.read()

    if ext.endswith(".zip"):
        from dol import zip_decompress

        table_src = zip_decompress(table_src)  # get decompressed bytes
        ext = ext[: -len(".zip")]  # remove the '.zip' from the extension

    return resolve_to_dataframe(
        table_src,
        ext=ext,
        ext_mapping=ext_mapping,
        **extra_decoder_kwargs,
    )


from tabled.wrappers import extension_based_encoding, extension_to_encoder


# TODO: Make the logic independent from local files assumption.
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
        *,
        extension_encoder_mapping: Dict[Extension, DfDecoder] = extension_to_encoder,
        extension_decoder_mapping: Dict[Extension, DfDecoder] = dflt_ext_mapping,
        extra_encoder_kwargs: Union[dict, Iterable] = (),
        extra_decoder_kwargs: Union[dict, Iterable] = (),
        allow_writing_bytes: bool = True,  # Note: can extend to validation callable
    ):
        super().__init__(rootdir)

        self.key_to_ext = file_extension

        self.extension_encoder_mapping = extension_encoder_mapping
        self.extension_decoder_mapping = extension_decoder_mapping

        extra_encoder_kwargs = dict(extra_encoder_kwargs)
        extra_decoder_kwargs = dict(extra_decoder_kwargs)

        self.bytes_and_ext_to_df = partial(
            df_from_data_given_ext,
            ext_mapping=extension_decoder_mapping,
            **extra_decoder_kwargs,
        )
        self.allow_writing_bytes = allow_writing_bytes

        # self.extension_based_encoder = partial(
        #     extension_based_encoding, extension_encoder_mapping
        # )

    def __getitem__(self, k):
        ext = self.key_to_ext(k)
        # data = BytesIO(super().__getitem__(k))
        data = super().__getitem__(k)
        return self.bytes_and_ext_to_df(data, ext)

    # TODO: Implement this. This works:
    # from tabled import extension_base_wrap
    # extension_base_wrap(Files)
    # But not this:
    def __setitem__(self, k, v):
        if not isinstance(v, bytes):
            bytes_ = extension_based_encoding(
                k, v, extension_to_encoder=self.extension_encoder_mapping
            )
        else:  # if v is bytes already
            if not self.allow_writing_bytes:
                raise ValueError("Cannot write bytes directly. Specify .")
            bytes_ = v
        # bytes_ = self.extension_based_encoding(v, file_extension(k))
        return super().__setitem__(k, bytes_)

    def __delitem__(self, k):
        return super().__delitem__(k)


class DfReader(DfFiles):
    def __setitem__(self, k, v):
        raise NotImplementedError("DfReader is a read-only store.")

    def __delitem__(self, k):
        raise NotImplementedError("DfReader is a read-only store.")


DfLocalFileReader = DfReader  # back-compatibility: TODO: Issue warning on use


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
