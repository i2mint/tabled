# tabled.base

Based functionality for tabled

### Functions

| [`convert_collection_to_dataframe_if_possible`](#tabled.base.convert_collection_to_dataframe_if_possible)(x)   | Return `x` as a DataFrame if it is a dict, list, tuple, Series or Index; else `x` unchanged.   |
|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [`get_table`](#tabled.base.get_table)([table_src, ext, ext_mapping, ...])    | Get a table from a variety of sources.                                                         |
| [`validate_fields`](#tabled.base.validate_fields)(df, key_fields, value_columns)   | Raise `ValueError` if any `key_fields` or `value_columns` are missing from `df`.               |

### Classes

| [`DataframeKvReader`](#tabled.base.DataframeKvReader)(df, key_fields[, ...])   | A Mapping view of a DataFrame, keyed by combinations of columns or index levels.   |
|---------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [`DfFiles`](#tabled.base.DfFiles)(rootdir, \*[, ...])                | A key-value store providing values as pandas.DataFrames.                           |
| [`DfLocalFileReader`](#tabled.base.DfLocalFileReader)                          |                                                                                    |
| [`DfReader`](#tabled.base.DfReader)(rootdir, \*[, ...])               | A read-only `DfFiles`: writes and deletes raise `NotImplementedError`.             |
| [`KeyFuncReader`](#tabled.base.KeyFuncReader)(mapping[, key])              | A read-only mapping view that transforms keys before lookup in `mapping`.          |

### *class* tabled.base.DataframeKvReader(df, key_fields, value_columns=None)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

A Mapping view of a DataFrame, keyed by combinations of columns or index levels.

* **Parameters:**
  * **df** – The DataFrame to wrap.
  * **key_fields** – Field(s) (columns or index levels) to use as keys.
  * **value_columns** – Column(s) to use as values. Defaults to all columns.

Example usage:

```pycon
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
>>> kv_reader[key].reset_index(drop=True)
   C   D
0  7  10
1  9  12
>>> list(kv_reader) == [(1, 4), (2, 5)]
True
```

But what if one (or more) of the key fields is an index level?
The DataframeKvReader can handle that too:

```pycon
>>> df = df.set_index(['A'])
>>> df
   B  C   D
A
1  4  7  10
2  5  8  11
1  4  9  12
>>> kv_reader = DataframeKvReader(df, ['A', 'B'], ['C', 'D'])
>>> key = (1, 4)
>>> kv_reader[key].reset_index(drop=True)
   C   D
0  7  10
1  9  12
>>> list(kv_reader) == [(1, 4), (2, 5)]
True
```

### *class* tabled.base.DfFiles(rootdir, \*, extension_encoder_mapping={'arrow': functools.partial(<function written_bytes>, <function dataframe_to_arrow_bytes>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'csv': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'dta': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_stata>, write_index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'feather': functools.partial(<function written_bytes>, <function DataFrame.to_feather>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'gbq': functools.partial(<function written_bytes>, <function \_to_gbq_unavailable>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'h5': functools.partial(<function written_bytes>, <function NDFrame.to_hdf>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'hdf5': functools.partial(<function written_bytes>, <function NDFrame.to_hdf>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'html': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_html>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'json': functools.partial(<function written_bytes>, <function NDFrame.to_json>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'npy': functools.partial(<function written_bytes>, <function save>, obj_arg_position_in_writer=1, io_buffer_cls=<class '_io.BytesIO'>), 'orc': functools.partial(<function written_bytes>, <function DataFrame.to_orc>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'p': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'parquet': functools.partial(<function written_bytes>, <function cast_to_parquet>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'pickle': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'pkl': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'sql': functools.partial(<function written_bytes>, <function NDFrame.to_sql>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'sqlite': functools.partial(<function written_bytes>, <function NDFrame.to_sql>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'stata': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_stata>, write_index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'tsv': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False, sep='\\\\t', escapechar='\\\\\\\\', quotechar='"'), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'txt': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xls': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_excel>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xlsx': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_excel>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xml': functools.partial(<function written_bytes>, <function DataFrame.to_xml>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'zip': <function save_df_to_zipped_tsv>}, extension_decoder_mapping={'arrow': functools.partial(<function read_from_bytes>, <function arrow_bytes_to_dataframe>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'csv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'dta': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'feather': functools.partial(<function read_from_bytes>, <function read_feather>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'h5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'hdf5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'html': functools.partial(<function read_from_bytes>, functools.partial(<function read_html>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'json': functools.partial(<function read_from_bytes>, functools.partial(<function read_json>, orient='records'), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'orc': functools.partial(<function read_from_bytes>, <function read_orc>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'p': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'parquet': functools.partial(<function read_from_bytes>, <function read_parquet>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pickle': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pkl': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sas': functools.partial(<function read_from_bytes>, <function read_sas>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sav': functools.partial(<function read_from_bytes>, <function read_spss>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sql': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sqlite': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'stata': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'tsv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, sep='\\\\t', index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'txt': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xls': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xlsx': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xml': functools.partial(<function read_from_bytes>, <function read_xml>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>)}, extra_encoder_kwargs=(), extra_decoder_kwargs=(), allow_writing_bytes=True, sqlite_tables=None, sqlite_verbose=False)

Bases: `Files`

A key-value store providing values as pandas.DataFrames.

Use Case: You have a bunch of files in a folder, all corresponding to some
dataframes that were saved in some way. You want to a key-value store whose values
are the (decoded) dataframes corresponding to the files in the folder.

Additionally, if you provide a SQLite database file instead of a directory,
it will automatically extract the tables as parquet files in a temporary directory
and provide access to them as DataFrames.

* **Parameters:**
  * **rootdir** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A root directory or a SQLite database file.
  * **extension_encoder_mapping** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Extension`), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Obj`)], `DataFrame`]]) – A mapping from file extensions to functions that
    encode a DataFrame to bytes for writing.
  * **extension_decoder_mapping** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Extension`), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Obj`)], `DataFrame`]]) – A mapping from file extensions to functions that can
    read the dataframes
  * **extra_encoder_kwargs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)) – Extra arguments to pass to the encoder functions.
  * **extra_decoder_kwargs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict) | [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)) – Extra arguments to pass to the decoder functions.
  * **allow_writing_bytes** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, values that are already `bytes` can be written
    as-is; if False, writing raw bytes raises a `ValueError`.
  * **sqlite_tables** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – When `rootdir` is a SQLite database file, the table names to
    export (all tables, if None).
  * **sqlite_verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – When `rootdir` is a SQLite database file, whether to print
    progress while exporting its tables.

#### *classmethod* from_sqlite_file(sqlite_file, , tables=None, verbose=False, \*\*kwargs)

Create a DfFiles instance from a SQLite database file.

This method exports all tables from the SQLite database to parquet files
in a temporary directory and returns a DfFiles instance that provides
access to these tables as DataFrames.

* **Parameters:**
  * **sqlite_file** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to the SQLite database file
  * **tables** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Sequence`](https://docs.python.org/3/library/typing.html#typing.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Optional list of table names to export. If None, exports all tables.
  * **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to print progress information
  * **\*\*kwargs** – Additional arguments passed to the DfFiles constructor
* **Return type:**
  [`DfFiles`](#tabled.base.DfFiles)
* **Returns:**
  A DfFiles instance providing access to the SQLite tables as DataFrames
* **Raises:**
  * [**FileNotFoundError**](https://docs.python.org/3/builtins/exceptions.html#FileNotFoundError) – If `sqlite_file` does not exist.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If `sqlite_file` does not look like a SQLite database.

### tabled.base.DfLocalFileReader

alias of [`DfReader`](#tabled.base.DfReader)

### *class* tabled.base.DfReader(rootdir, \*, extension_encoder_mapping={'arrow': functools.partial(<function written_bytes>, <function dataframe_to_arrow_bytes>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'csv': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'dta': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_stata>, write_index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'feather': functools.partial(<function written_bytes>, <function DataFrame.to_feather>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'gbq': functools.partial(<function written_bytes>, <function \_to_gbq_unavailable>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'h5': functools.partial(<function written_bytes>, <function NDFrame.to_hdf>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'hdf5': functools.partial(<function written_bytes>, <function NDFrame.to_hdf>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'html': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_html>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'json': functools.partial(<function written_bytes>, <function NDFrame.to_json>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'npy': functools.partial(<function written_bytes>, <function save>, obj_arg_position_in_writer=1, io_buffer_cls=<class '_io.BytesIO'>), 'orc': functools.partial(<function written_bytes>, <function DataFrame.to_orc>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'p': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'parquet': functools.partial(<function written_bytes>, <function cast_to_parquet>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'pickle': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'pkl': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'sql': functools.partial(<function written_bytes>, <function NDFrame.to_sql>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'sqlite': functools.partial(<function written_bytes>, <function NDFrame.to_sql>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'stata': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_stata>, write_index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'tsv': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False, sep='\\\\t', escapechar='\\\\\\\\', quotechar='"'), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'txt': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xls': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_excel>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xlsx': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_excel>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xml': functools.partial(<function written_bytes>, <function DataFrame.to_xml>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'zip': <function save_df_to_zipped_tsv>}, extension_decoder_mapping={'arrow': functools.partial(<function read_from_bytes>, <function arrow_bytes_to_dataframe>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'csv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'dta': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'feather': functools.partial(<function read_from_bytes>, <function read_feather>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'h5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'hdf5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'html': functools.partial(<function read_from_bytes>, functools.partial(<function read_html>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'json': functools.partial(<function read_from_bytes>, functools.partial(<function read_json>, orient='records'), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'orc': functools.partial(<function read_from_bytes>, <function read_orc>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'p': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'parquet': functools.partial(<function read_from_bytes>, <function read_parquet>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pickle': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pkl': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sas': functools.partial(<function read_from_bytes>, <function read_sas>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sav': functools.partial(<function read_from_bytes>, <function read_spss>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sql': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sqlite': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'stata': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'tsv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, sep='\\\\t', index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'txt': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xls': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xlsx': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xml': functools.partial(<function read_from_bytes>, <function read_xml>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>)}, extra_encoder_kwargs=(), extra_decoder_kwargs=(), allow_writing_bytes=True, sqlite_tables=None, sqlite_verbose=False)

Bases: [`DfFiles`](#tabled.base.DfFiles)

A read-only `DfFiles`: writes and deletes raise `NotImplementedError`.

### *class* tabled.base.KeyFuncReader(mapping, key=<function identity>)

Bases: `KvReader`

A read-only mapping view that transforms keys before lookup in `mapping`.

Iteration and length reflect `mapping` as-is; `__getitem__` and
`__contains__` apply `key` to the given key first.

### tabled.base.convert_collection_to_dataframe_if_possible(x)

Return `x` as a DataFrame if it is a dict, list, tuple, Series or Index; else `x` unchanged.

### tabled.base.get_table(table_src=None, \*, ext=None, ext_mapping={'arrow': functools.partial(<function read_from_bytes>, <function arrow_bytes_to_dataframe>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'csv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'dta': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'feather': functools.partial(<function read_from_bytes>, <function read_feather>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'h5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'hdf5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'html': functools.partial(<function read_from_bytes>, functools.partial(<function read_html>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'json': functools.partial(<function read_from_bytes>, functools.partial(<function read_json>, orient='records'), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'orc': functools.partial(<function read_from_bytes>, <function read_orc>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'p': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'parquet': functools.partial(<function read_from_bytes>, <function read_parquet>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pickle': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pkl': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sas': functools.partial(<function read_from_bytes>, <function read_sas>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sav': functools.partial(<function read_from_bytes>, <function read_spss>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sql': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sqlite': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'stata': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'tsv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, sep='\\\\t', index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'txt': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xls': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xlsx': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xml': functools.partial(<function read_from_bytes>, <function read_xml>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>)}, resolve_to_io=<function default_io_resolver>, \*\*extra_decoder_kwargs)

Get a table from a variety of sources.

* **Return type:**
  `DataFrame`

### tabled.base.validate_fields(df, key_fields, value_columns)

Raise `ValueError` if any `key_fields` or `value_columns` are missing from `df`.
