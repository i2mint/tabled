# tabled.compare_tables

Tools to compare tables

### Functions

| [`columns_diff`](#tabled.compare_tables.columns_diff)(df1, df2)                        | Return columns that are not common between df1 and df2.                   |
|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`columns_value_diff`](#tabled.compare_tables.columns_value_diff)(df1, df2)                  | For each column present in both dataframes, compare the values row-wise.  |
| [`dataframe_diffs`](#tabled.compare_tables.dataframe_diffs)(df1, df2[, comparisons, ...]) | Compare the diff of dataframes using specified diff comparison functions. |
| [`dtypes_diff`](#tabled.compare_tables.dtypes_diff)(df1, df2)                         | Return columns where the data types differ between df1 and df2.           |
| [`ensure_comparisons_dict`](#tabled.compare_tables.ensure_comparisons_dict)(comparisons)          | Ensure that the comparisons are in the form of a dictionary.              |
| [`index_diff`](#tabled.compare_tables.index_diff)(df1, df2)                          | Return indices that are not common between df1 and df2.                   |
| [`shape_diff`](#tabled.compare_tables.shape_diff)(df1, df2)                          | Return the shapes of df1 and df2 if they differ.                          |

### Classes

| [`BinaryFuncResult`](#tabled.compare_tables.BinaryFuncResult)   | A `{left_right, right_left}` dict, truthy if either value is truthy.   |
|---------------------------------------------------------------------|------------------------------------------------------------------------|

### Exceptions

| [`InvalidComparison`](#tabled.compare_tables.InvalidComparison)   | Raised or returned when a comparison that was asked for is not applicable to the dataframes in question   |
|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|

### *class* tabled.compare_tables.BinaryFuncResult

Bases: [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

A `{left_right, right_left}` dict, truthy if either value is truthy.

#### *classmethod* from_func(func, x, y)

Build a `BinaryFuncResult` from `func(x, y)` and `func(y, x)`.

### *exception* tabled.compare_tables.InvalidComparison

Bases: [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

Raised or returned when a comparison that was asked for is not applicable to the dataframes in question

### tabled.compare_tables.columns_diff(df1, df2)

Return columns that are not common between df1 and df2.

### tabled.compare_tables.columns_value_diff(df1, df2)

For each column present in both dataframes, compare the values row-wise.
Returns a dict with column names as keys and DataFrames of differences as values.
Only columns with differences are included.

### tabled.compare_tables.dataframe_diffs(df1, df2, comparisons={'columns_diff': <function columns_diff>, 'columns_value_diff': <function columns_value_diff>, 'dtypes_diff': <function dtypes_diff>, 'index_diff': <function index_diff>, 'shape_diff': <function shape_diff>}, \*, diff_condition=<class 'bool'>)

Compare the diff of dataframes using specified diff comparison functions.

Returns a dictionary with comparison names as keys and comparison results as values.

* **Parameters:**
  * **df1** (`DataFrame`) – The first dataframe to compare.
  * **df2** (`DataFrame`) – The second dataframe to compare.
  * **comparisons** (`Union`[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`DataFrame`, `DataFrame`], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]], [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`DataFrame`, `DataFrame`], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`DataFrame`, `DataFrame`], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – A dictionary or list of comparison functions or names.
    Defaults to DFLT_COMPARISONS.
  * **diff_condition** (*callable*) – A function that determines whether to include a comparison result
    in the output dictionary based on the comparison result.
    Defaults to the built-in `bool` function.
* **Returns:**
  A dictionary with comparison names as keys and comparison results as values.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Example

```pycon
>>> import pandas as pd
>>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=[0, 1])
>>> df2 = pd.DataFrame({'A': [1, 2], 'C': [5, 6]}, index=[1, 2])
>>> diffs = dataframe_diffs(df1, df2)
>>> diffs
{'columns_diff': {'left_right': {'B'}, 'right_left': {'C'}},
 'index_diff': {'left_right': {0}, 'right_left': {2}},
 'columns_value_diff': {'A':    left  right
1     2      1}}
```

### tabled.compare_tables.dtypes_diff(df1, df2)

Return columns where the data types differ between df1 and df2.

### tabled.compare_tables.ensure_comparisons_dict(comparisons)

Ensure that the comparisons are in the form of a dictionary.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`DataFrame`, `DataFrame`], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### tabled.compare_tables.index_diff(df1, df2)

Return indices that are not common between df1 and df2.

### tabled.compare_tables.shape_diff(df1, df2)

Return the shapes of df1 and df2 if they differ.
