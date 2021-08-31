"""
A  (key-value) data-object-layer to get (pandas) tables from a variety of sources with ease.


"""
from functools import partial
from io import BytesIO
import os
import pickle

import pandas as pd

from py2store.stores.local_store import LocalBinaryStore

from tabled.html import (
    url_to_html_func,
    get_tables_from_url,
    dfs_to_html_pretty,
    dfs_to_pdf_bytes,
)

DFLT_EXT_SPECS = {}


def df_from_data_given_ext(data, ext, ext_specs=None, **kwargs):
    """Get a dataframe from a (data, ext) pair"""
    ext_specs = ext_specs or DFLT_EXT_SPECS  # NOTE: Note used yet
    if ext.startswith("."):
        ext = ext[1:]
    if ext in {"xls", "xlsx"}:
        kwargs = dict({"index": False}, **kwargs)
        return pd.read_excel(data, **kwargs)
    elif ext in {"csv"}:
        kwargs = dict({"index_col": False}, **kwargs)
        return pd.read_csv(data, **kwargs)
    elif ext in {"tsv"}:
        kwargs = dict({"sep": "\t", "index_col": False}, **kwargs)
        return pd.read_csv(data, **kwargs)
    elif ext in {"json"}:
        kwargs = dict({"orient": "records"}, **kwargs)
        return pd.read_json(data, **kwargs)
    elif ext in {"html"}:
        kwargs = dict({"index_col": False}, **kwargs)
        return pd.read_html(data, **kwargs)[0]
    elif ext in {"p", "pickle"}:
        return pickle.load(data, **kwargs)
    else:
        raise ValueError(f"Don't know how to handle extension: {ext}")


# TODO: Make the logic independent from local files assumption.
# TODO: Better separate Reader, and add DfStore to make a writer.


class DfLocalFileReader(LocalBinaryStore):
    """A key-value store providing values as pandas.DataFrames"""

    def __init__(self, path_format, ext_specs=None):
        super().__init__(path_format)
        if ext_specs is None:
            ext_specs = DFLT_EXT_SPECS
        self._ext_specs = ext_specs
        self.data_and_ext_to_df = partial(df_from_data_given_ext, ext_specs=ext_specs)
        self.data_and_ext_to_df = (
            df_from_data_given_ext  # TODO: Hard coded for now, to keep functioning
        )

    def __getitem__(self, k):
        ext = self.key_to_ext(k)
        kwargs = self.ext_specs.get(ext, {})
        data = BytesIO(super().__getitem__(k))
        return df_from_data_given_ext(data, ext, **kwargs)

    def key_to_ext(self, k):
        _, ext = os.path.splitext(k)
        if ext.startswith("."):
            ext = ext[1:]
        return ext

    def __setitem__(self, k, v):
        raise NotImplementedError("This is a reader: No write operation allowed")

    def __delitem__(self, k):
        raise NotImplementedError("This is a reader: No delete operation allowed")


DfReader = DfLocalFileReader  # alias for back-compatibility: TODO: Issue warning on use
