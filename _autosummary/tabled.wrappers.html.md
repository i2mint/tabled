# tabled.wrappers

Wrapping tools

A lot of what is defined here are functions that are used to transform data.
More precisely, encode and decode data depending on it’s format, file extension, etc.

### Functions

| [`add_extension_codec`](#tabled.wrappers.add_extension_codec)([extension, encoder, ...])      | Add an extension-based encoder and decoder to the extension-code mapping.                  |
|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| [`cast_to_parquet`](#tabled.wrappers.cast_to_parquet)(data, \*args[, \_\_name_of_column]) | Convert data to DataFrame if necessary, then save as parquet.                              |
| [`default_io_resolver`](#tabled.wrappers.default_io_resolver)(src)                            | Resolve `src` (a local path, an http(s)/graze URL, or bytes) to a binary file-like object. |
| [`df_from_data_according_to_key`](#tabled.wrappers.df_from_data_according_to_key)(data, mapping, ...)   | Get a dataframe from a (data, mapping, key) triple                                         |
| [`df_from_data_given_ext`](#tabled.wrappers.df_from_data_given_ext)(data, ext[, ext_mapping])    | Get a dataframe from a (data, ext) pair                                                    |
| [`extension_based_decoding`](#tabled.wrappers.extension_based_decoding)(k, v, \*[, ...])           | Decode a value based on the extension of the key.                                          |
| [`extension_based_encoding`](#tabled.wrappers.extension_based_encoding)(k, v, \*[, ...])           | Encode a value based on the extension of the key.                                          |
| [`extension_based_wrap`](#tabled.wrappers.extension_based_wrap)([store, ...])                  | Add extension-based encoding and decoding to a store.                                      |
| [`file_extension`](#tabled.wrappers.file_extension)(key)                                 | Get the file extension from a key                                                          |
| [`get_codec_mappings`](#tabled.wrappers.get_codec_mappings)(\*[, ...])                       | Return `{"encoders": extension_to_encoder, "decoders": extension_to_decoder}`.             |
| [`get_extension`](#tabled.wrappers.get_extension)(key)                                  | Return the extension of a file path.                                                       |
| [`get_file_ext`](#tabled.wrappers.get_file_ext)(key)                                   | Get the file extension from a key                                                          |
| [`get_protocol`](#tabled.wrappers.get_protocol)(url)                                   | Get the protocol of a url                                                                  |
| [`if_extension_not_present_add_it`](#tabled.wrappers.if_extension_not_present_add_it)(filepath, ...)      | Append `extension` to `filepath` unless it's already there.                                |
| [`if_extension_present_remove_it`](#tabled.wrappers.if_extension_present_remove_it)(filepath, ...)       | Strip a trailing `extension` from `filepath`, if present.                                  |
| [`key_func_mapping`](#tabled.wrappers.key_func_mapping)(obj, mapping[, key, ...])          | Map an object to a value based on a key function                                           |
| [`map_values`](#tabled.wrappers.map_values)(func, d, \*[, except_condition, ...])    | Map values of a dictionary, except for those that satisfy a condition.                     |
| [`print_current_mappings`](#tabled.wrappers.print_current_mappings)()                            | Print the current extension-to-encoder and extension-to-decoder mappings.                  |
| [`resolve_to_dataframe`](#tabled.wrappers.resolve_to_dataframe)(data, ext[, ext_mapping])      | Get a dataframe from a (data, ext) pair                                                    |
| [`save_df_to_zipped_tsv`](#tabled.wrappers.save_df_to_zipped_tsv)(df, name[, sep, index])       | Save a dataframe to a zipped tsv file.                                                     |
| [`single_column_parquet_decode`](#tabled.wrappers.single_column_parquet_decode)(b[, col])              | Decode a single-column parquet file into a list of sequences.                              |
| [`single_column_parquet_encode`](#tabled.wrappers.single_column_parquet_encode)(sequences[, col])      | Encode a list of sequences into a single-column parquet file.                              |

### tabled.wrappers.add_extension_codec(extension=None, , encoder=None, decoder=None, overwrite=False)

Add an extension-based encoder and decoder to the extension-code mapping.

Sure, you could just edit the underlying dictionaries directly, but the design gods
would not be pleased.

If no arguments are passed, it will print the current mappings.

* **Parameters:**
  * **extension** – The file extension to add the codec for. If None, it will print the current mappings.
  * **encoder** – The encoder function to add. If None, it will print the current mappings.
  * **decoder** – The decoder function to add. If None, it will print the current mappings.
  * **overwrite** – If True, it will overwrite the existing encoder/decoder for the given extension.
    If False, it will raise a ValueError if the extension already exists.
* **Returns:**
  None. It just adds to the in-memory mappings (or prints them).

### tabled.wrappers.cast_to_parquet(data, \*args, \_\_name_of_column='_\_single_column_values', \*\*kwargs)

Convert data to DataFrame if necessary, then save as parquet.

Handles:

- pandas.DataFrame: use as-is
- pandas.Series: convert to DataFrame using to_frame()
- list/other iterables: convert to Series then DataFrame

### tabled.wrappers.default_io_resolver(src)

Resolve `src` (a local path, an http(s)/graze URL, or bytes) to a binary file-like object.

* **Return type:**
  [`BinaryIO`](https://docs.python.org/3/library/typing.html#typing.BinaryIO)

### tabled.wrappers.df_from_data_according_to_key(data, mapping, key, \*\*extra_decoder_kwargs)

Get a dataframe from a (data, mapping, key) triple

### tabled.wrappers.df_from_data_given_ext(data, ext, ext_mapping={'arrow': functools.partial(<function read_from_bytes>, <function arrow_bytes_to_dataframe>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'csv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'dta': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'feather': functools.partial(<function read_from_bytes>, <function read_feather>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'h5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'hdf5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'html': functools.partial(<function read_from_bytes>, functools.partial(<function read_html>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'json': functools.partial(<function read_from_bytes>, functools.partial(<function read_json>, orient='records'), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'orc': functools.partial(<function read_from_bytes>, <function read_orc>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'p': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'parquet': functools.partial(<function read_from_bytes>, <function read_parquet>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pickle': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pkl': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sas': functools.partial(<function read_from_bytes>, <function read_sas>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sav': functools.partial(<function read_from_bytes>, <function read_spss>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sql': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sqlite': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'stata': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'tsv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, sep='\\\\t', index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'txt': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xls': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xlsx': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xml': functools.partial(<function read_from_bytes>, <function read_xml>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>)}, \*\*extra_decoder_kwargs)

Get a dataframe from a (data, ext) pair

* **Return type:**
  `DataFrame`

### tabled.wrappers.extension_based_decoding(k, v, \*, extension_to_decoder={'arrow': functools.partial(<function read_from_bytes>, <function arrow_bytes_to_dataframe>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'csv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'dta': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'feather': functools.partial(<function read_from_bytes>, <function read_feather>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'h5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'hdf5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'html': functools.partial(<function read_from_bytes>, functools.partial(<function read_html>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'json': functools.partial(<function read_from_bytes>, functools.partial(<function read_json>, orient='records'), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'orc': functools.partial(<function read_from_bytes>, <function read_orc>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'p': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'parquet': functools.partial(<function read_from_bytes>, <function read_parquet>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pickle': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pkl': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sas': functools.partial(<function read_from_bytes>, <function read_sas>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sav': functools.partial(<function read_from_bytes>, <function read_spss>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sql': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sqlite': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'stata': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'tsv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, sep='\\t', index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'txt': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xls': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xlsx': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xml': functools.partial(<function read_from_bytes>, <function read_xml>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>)})

Decode a value based on the extension of the key.

### tabled.wrappers.extension_based_encoding(k, v, \*, extension_to_encoder={'arrow': functools.partial(<function written_bytes>, <function dataframe_to_arrow_bytes>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'csv': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'dta': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_stata>, write_index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'feather': functools.partial(<function written_bytes>, <function DataFrame.to_feather>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'gbq': functools.partial(<function written_bytes>, <function \_to_gbq_unavailable>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'h5': functools.partial(<function written_bytes>, <function NDFrame.to_hdf>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'hdf5': functools.partial(<function written_bytes>, <function NDFrame.to_hdf>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'html': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_html>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'json': functools.partial(<function written_bytes>, <function NDFrame.to_json>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'npy': functools.partial(<function written_bytes>, <function save>, obj_arg_position_in_writer=1, io_buffer_cls=<class '_io.BytesIO'>), 'orc': functools.partial(<function written_bytes>, <function DataFrame.to_orc>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'p': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'parquet': functools.partial(<function written_bytes>, <function cast_to_parquet>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'pickle': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'pkl': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'sql': functools.partial(<function written_bytes>, <function NDFrame.to_sql>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'sqlite': functools.partial(<function written_bytes>, <function NDFrame.to_sql>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'stata': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_stata>, write_index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'tsv': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False, sep='\\t', escapechar='\\\\', quotechar='"'), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'txt': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xls': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_excel>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xlsx': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_excel>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xml': functools.partial(<function written_bytes>, <function DataFrame.to_xml>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'zip': <function save_df_to_zipped_tsv>})

Encode a value based on the extension of the key.

### tabled.wrappers.extension_based_wrap(store=None, \*, extension_to_decoder={'arrow': functools.partial(<function read_from_bytes>, <function arrow_bytes_to_dataframe>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'csv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'dta': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'feather': functools.partial(<function read_from_bytes>, <function read_feather>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'h5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'hdf5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'html': functools.partial(<function read_from_bytes>, functools.partial(<function read_html>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'json': functools.partial(<function read_from_bytes>, functools.partial(<function read_json>, orient='records'), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'orc': functools.partial(<function read_from_bytes>, <function read_orc>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'p': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'parquet': functools.partial(<function read_from_bytes>, <function read_parquet>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pickle': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pkl': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sas': functools.partial(<function read_from_bytes>, <function read_sas>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sav': functools.partial(<function read_from_bytes>, <function read_spss>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sql': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sqlite': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'stata': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'tsv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, sep='\\t', index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'txt': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xls': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xlsx': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xml': functools.partial(<function read_from_bytes>, <function read_xml>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>)}, extension_to_encoder={'arrow': functools.partial(<function written_bytes>, <function dataframe_to_arrow_bytes>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'csv': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'dta': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_stata>, write_index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'feather': functools.partial(<function written_bytes>, <function DataFrame.to_feather>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'gbq': functools.partial(<function written_bytes>, <function \_to_gbq_unavailable>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'h5': functools.partial(<function written_bytes>, <function NDFrame.to_hdf>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'hdf5': functools.partial(<function written_bytes>, <function NDFrame.to_hdf>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'html': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_html>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'json': functools.partial(<function written_bytes>, <function NDFrame.to_json>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'npy': functools.partial(<function written_bytes>, <function save>, obj_arg_position_in_writer=1, io_buffer_cls=<class '_io.BytesIO'>), 'orc': functools.partial(<function written_bytes>, <function DataFrame.to_orc>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'p': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'parquet': functools.partial(<function written_bytes>, <function cast_to_parquet>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'pickle': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'pkl': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'sql': functools.partial(<function written_bytes>, <function NDFrame.to_sql>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'sqlite': functools.partial(<function written_bytes>, <function NDFrame.to_sql>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'stata': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_stata>, write_index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'tsv': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False, sep='\\t', escapechar='\\\\', quotechar='"'), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'txt': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xls': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_excel>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xlsx': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_excel>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xml': functools.partial(<function written_bytes>, <function DataFrame.to_xml>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'zip': <function save_df_to_zipped_tsv>}, \_\_module_\_=None, \_\_name_\_=None, \_\_qualname_\_=None, \_\_doc_\_=None, \_\_annotations_\_=None, \_\_defaults_\_=None, \_\_kwdefaults_\_=None)

Add extension-based encoding and decoding to a store.

### tabled.wrappers.file_extension(key)

Get the file extension from a key

* **Return type:**
  [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Extension`)

```pycon
>>> file_extension('hello.world')
'world'
>>> file_extension('hello')
''
```

### tabled.wrappers.get_codec_mappings(\*, extension_to_encoder={'arrow': functools.partial(<function written_bytes>, <function dataframe_to_arrow_bytes>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'csv': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'dta': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_stata>, write_index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'feather': functools.partial(<function written_bytes>, <function DataFrame.to_feather>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'gbq': functools.partial(<function written_bytes>, <function \_to_gbq_unavailable>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'h5': functools.partial(<function written_bytes>, <function NDFrame.to_hdf>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'hdf5': functools.partial(<function written_bytes>, <function NDFrame.to_hdf>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'html': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_html>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'json': functools.partial(<function written_bytes>, <function NDFrame.to_json>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'npy': functools.partial(<function written_bytes>, <function save>, obj_arg_position_in_writer=1, io_buffer_cls=<class '_io.BytesIO'>), 'orc': functools.partial(<function written_bytes>, <function DataFrame.to_orc>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'p': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'parquet': functools.partial(<function written_bytes>, <function cast_to_parquet>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'pickle': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'pkl': functools.partial(<function written_bytes>, <function NDFrame.to_pickle>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'sql': functools.partial(<function written_bytes>, <function NDFrame.to_sql>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'sqlite': functools.partial(<function written_bytes>, <function NDFrame.to_sql>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'stata': functools.partial(<function written_bytes>, functools.partial(<function DataFrame.to_stata>, write_index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'tsv': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False, sep='\\t', escapechar='\\\\', quotechar='"'), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'txt': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_csv>, index=False), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xls': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_excel>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xlsx': functools.partial(<function written_bytes>, functools.partial(<function NDFrame.to_excel>, index=True), obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'xml': functools.partial(<function written_bytes>, <function DataFrame.to_xml>, obj_arg_position_in_writer=0, io_buffer_cls=<class '_io.BytesIO'>), 'zip': <function save_df_to_zipped_tsv>}, extension_to_decoder={'arrow': functools.partial(<function read_from_bytes>, <function arrow_bytes_to_dataframe>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'csv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'dta': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'feather': functools.partial(<function read_from_bytes>, <function read_feather>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'h5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'hdf5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'html': functools.partial(<function read_from_bytes>, functools.partial(<function read_html>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'json': functools.partial(<function read_from_bytes>, functools.partial(<function read_json>, orient='records'), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'orc': functools.partial(<function read_from_bytes>, <function read_orc>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'p': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'parquet': functools.partial(<function read_from_bytes>, <function read_parquet>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pickle': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pkl': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sas': functools.partial(<function read_from_bytes>, <function read_sas>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sav': functools.partial(<function read_from_bytes>, <function read_spss>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sql': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sqlite': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'stata': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'tsv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, sep='\\t', index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'txt': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xls': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xlsx': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xml': functools.partial(<function read_from_bytes>, <function read_xml>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>)})

Return `{"encoders": extension_to_encoder, "decoders": extension_to_decoder}`.

### tabled.wrappers.get_extension(key)

Return the extension of a file path.

Note that it includes the dot.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> get_extension('hello.world')
'.world'
```

If there’s no extension, it returns an empty string.

```pycon
>>> get_extension('hello')
''
```

### tabled.wrappers.get_file_ext(key)

Get the file extension from a key

* **Return type:**
  [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Extension`)

```pycon
>>> file_extension('hello.world')
'world'
>>> file_extension('hello')
''
```

### tabled.wrappers.get_protocol(url)

Get the protocol of a url

```pycon
>>> get_protocol('https://www.google.com')
'https'
>>> get_protocol('file:///home/user/file.txt')
'file'
```

The function returns None if no protocol is found:

```pycon
>>> assert get_protocol('no_protocol_here') is None
```

### tabled.wrappers.if_extension_not_present_add_it(filepath, extension)

Append `extension` to `filepath` unless it’s already there.

### tabled.wrappers.if_extension_present_remove_it(filepath, extension)

Strip a trailing `extension` from `filepath`, if present.

### tabled.wrappers.key_func_mapping(obj, mapping, key=<function identity>, not_found_sentinel=Sentinel('dflt_not_found_sentinel'))

Map an object to a value based on a key function

* **Return type:**
  [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`)

### tabled.wrappers.map_values(func, d, \*, except_condition=functools.partial(<function \_isinstance>, class_or_tuple=<class 'i2.util.LiteralVal'>), except_handler=operator.methodcaller('_\_call_\_'))

Map values of a dictionary, except for those that satisfy a condition.

The `except_condition` is a function that takes a value and returns a boolean.
If the condition is True, the value is not mapped.
Instead, the `except_handler` is called with the value, and the result is used as
the new value (often, the value is left unchanged).

The default `except_condition` is `is_instance_of(LiteralVal)`, which is a function
that returns True if the value is to be taken litterally.
The default `except_handler` is `methodcaller('__call__')`, which will extract
the litteral value from the `LiteralVal` object.

```pycon
>>> map_values(lambda x: x * 10, {'a': 1, 'b': LiteralVal(2), 'c': 3})
{'a': 10, 'b': 2, 'c': 30}
```

### tabled.wrappers.print_current_mappings()

Print the current extension-to-encoder and extension-to-decoder mappings.

### tabled.wrappers.resolve_to_dataframe(data, ext, ext_mapping={'arrow': functools.partial(<function read_from_bytes>, <function arrow_bytes_to_dataframe>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'csv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'dta': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'feather': functools.partial(<function read_from_bytes>, <function read_feather>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'h5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'hdf5': functools.partial(<function read_from_bytes>, <function read_hdf>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'html': functools.partial(<function read_from_bytes>, functools.partial(<function read_html>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'json': functools.partial(<function read_from_bytes>, functools.partial(<function read_json>, orient='records'), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'orc': functools.partial(<function read_from_bytes>, <function read_orc>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'p': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'parquet': functools.partial(<function read_from_bytes>, <function read_parquet>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pickle': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'pkl': functools.partial(<function read_from_bytes>, <built-in function load>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sas': functools.partial(<function read_from_bytes>, <function read_sas>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sav': functools.partial(<function read_from_bytes>, <function read_spss>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sql': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'sqlite': functools.partial(<function read_from_bytes>, <function read_sql>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'stata': functools.partial(<function read_from_bytes>, functools.partial(<function read_stata>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'tsv': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, sep='\\\\t', index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'txt': functools.partial(<function read_from_bytes>, functools.partial(<function read_csv>, index_col=None), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xls': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xlsx': functools.partial(<function read_from_bytes>, functools.partial(<function read_excel>, index_col=0), buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>), 'xml': functools.partial(<function read_from_bytes>, <function read_xml>, buffer_arg_position=0, buffer_arg_name=None, io_buffer_cls=<class '_io.BytesIO'>)}, \*\*extra_decoder_kwargs)

Get a dataframe from a (data, ext) pair

* **Return type:**
  `DataFrame`

### tabled.wrappers.save_df_to_zipped_tsv(df, name, sep='\\\\t', index=False, \*\*kwargs)

Save a dataframe to a zipped tsv file.

### tabled.wrappers.single_column_parquet_decode(b, col='_\_single_column_values')

Decode a single-column parquet file into a list of sequences.

#### SEE ALSO
single_column_parquet_encode

```pycon
>>> sequences_2 = [['one', 'two'], ['three', 'four', 'five']]
>>> encoded_2 = single_column_parquet_encode(sequences_2)
>>> decoded_2 = single_column_parquet_decode(encoded_2)
>>> all((x == y).all() for x, y in zip(decoded_2, sequences_2))
True
```

### tabled.wrappers.single_column_parquet_encode(sequences, col='_\_single_column_values')

Encode a list of sequences into a single-column parquet file.
See more general function: cast_to_parquet.

The raison d’etre of this function is to have a two-way codec for sequences->parquet

```pycon
>>> sequences_1 = [[1, 2], [3, 4, 5]]
>>> encoded_1 = single_column_parquet_encode(sequences_1)
>>> decoded_1 = single_column_parquet_decode(encoded_1)
>>> all((x == y).all() for x, y in zip(decoded_1, sequences_1))
True
```
