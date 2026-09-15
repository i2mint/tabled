# tabled.diagnose

DataFrame and table collection diagnosis utilities.

This module provides flexible tools for analyzing pandas DataFrames and collections
of tables. The core function `dataframe_info` extracts configurable information
from DataFrames using pluggable info functions.

Key Features:

- Configurable info extraction with `dataframe_info`
- Collection diagnosis with `diagnose_table_collection`
- Extensible via custom info functions
- Backward-compatible `print_dataframe_info` from cosmodata

### Example

```pycon
>>> import pandas as pd
>>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
>>> info = dataframe_info(df)
>>> info['shape']
(3, 2)
```

### Register custom info function

```pycon
>>> def get_memory_usage(df):
...     return df.memory_usage(deep=True).sum()
>>> register_info_func('custom_memory', get_memory_usage)
```

### Functions

| [`dataframe_info`](#tabled.diagnose.dataframe_info)(df[, info_funcs, egress])         | Extract information from a DataFrame using specified info functions.        |
|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`list_info_funcs`](#tabled.diagnose.list_info_funcs)()                                | List all registered info function names.                                    |
| [`print_dataframe_info`](#tabled.diagnose.print_dataframe_info)(df[, exclude_columns, ...]) | Print information about a DataFrame.                                        |
| [`register_info_func`](#tabled.diagnose.register_info_func)(name, func, \*[, overwrite])  | Register a new info function in the default info functions dictionary.      |
| [`scalar_columns`](#tabled.diagnose.scalar_columns)(df)                               | Returns the list of columns that are scalar (therefore serializable to CSV) |

### tabled.diagnose.dataframe_info(df, info_funcs={'categorical_stats': <function \_get_categorical_stats>, 'columns': <function \_get_columns>, 'first_row': <function \_get_first_row>, 'numeric_stats': <function \_get_numeric_stats>, 'sample_rows': <function \_get_sample_rows>, 'shape': <function \_get_shape>}, \*, egress=<class 'dict'>)

Extract information from a DataFrame using specified info functions.

* **Parameters:**
  * **df** (`DataFrame`) – The DataFrame to analyze
  * **info_funcs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Dict mapping info keys to functions that take a DataFrame
  * **egress** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to process the generator of (key, value) pairs
* **Returns:**
  Result of egress applied to the info generator

```pycon
>>> import pandas as pd
>>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
>>> info = dataframe_info(df, {'shape': _get_shape})
>>> info['shape']
(3, 2)
```

### tabled.diagnose.list_info_funcs()

List all registered info function names.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### tabled.diagnose.print_dataframe_info(df, exclude_columns=(), \*, mode='short', egress=<built-in function print>)

Print information about a DataFrame.

* **Parameters:**
  * **df** (`DataFrame`) – The DataFrame to analyze
  * **exclude_columns** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Columns to exclude from analysis
  * **mode** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'short'`, `'sample'`, `'stats'`]) – 

    Type of information to display
    - ’short’: shape and first row
    - ’sample’: shape, columns, and random rows
    - ’stats’: descriptive statistics
  * **egress** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`None`](https://docs.python.org/3/builtins/constants.html#None)]]) – Callback function for output (None returns string instead of printing)
* **Returns:**
  The formatted info string when `egress` is `None` or falsy; otherwise
  the result of calling `egress` on that string.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If `mode` is not one of `'short'`, `'sample'`, `'stats'`.

```pycon
>>> import pandas as pd
>>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
>>> info = print_dataframe_info(df, egress=None)
>>> 'shape: (3, 2)' in info
True
```

### tabled.diagnose.register_info_func(name, func, , overwrite=False)

Register a new info function in the default info functions dictionary.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Name for the info function
  * **func** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`DataFrame`], `any`]) – Function that takes a DataFrame and returns info
  * **overwrite** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to overwrite existing functions with the same name
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If `name` is already registered and `overwrite` is `False`.

### Example

```pycon
>>> def get_memory_usage(df):
...     return df.memory_usage(deep=True).sum()
>>> register_info_func('test_memory', get_memory_usage)
```

### tabled.diagnose.scalar_columns(df)

Returns the list of columns that are scalar (therefore serializable to CSV)

More precisely, this function returns the list of columns that contain only
scalar values (e.g., int, float, str, bool, etc.) and can be saved to a CSV
file.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

### Example

```pycon
>>> import pandas as pd
>>> df = pd.DataFrame({
...     'A': [1, 2, 3],
...     'B': ['x', 'y', 'z'],
...     'C': [{'a': 1}, {'b': 2}, {'c': 3}],  # Non-serializable column
...     'D': [[1, 2], [3, 4], [5, 6]]         # Non-serializable column
... })
>>> scalar_columns(df)
['A', 'B']
```
