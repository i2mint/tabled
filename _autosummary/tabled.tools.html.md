# tabled.tools

Various high-level tools using tabled

### Functions

| [`diagnose_table_collection`](#tabled.tools.diagnose_table_collection)(tables, \*[, ...])   | Diagnose a collection of tables and return diagnostic information.   |
|-------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|

### tabled.tools.diagnose_table_collection(tables, \*, info_funcs={'categorical_stats': <function \_get_categorical_stats>, 'columns': <function \_get_columns>, 'first_row': <function \_get_first_row>, 'numeric_stats': <function \_get_numeric_stats>, 'sample_rows': <function \_get_sample_rows>, 'shape': <function \_get_shape>}, egress=<class 'dict'>)

Diagnose a collection of tables and return diagnostic information.

* **Parameters:**
  * **tables** – 

    Collection of tables - can be:
    - A mapping from keys to DataFrames
    - A non-mapping, non-string iterable of DataFrames (will use enumerate for keys)
    - A string URI to create DfFiles mapping
  * **info_funcs** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)]) – Dictionary of info functions to apply
  * **egress** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – Function to process the generator of (table_key, info_dict) pairs
* **Returns:**
  Result of egress applied to the info generator
* **Raises:**
  [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) – If `tables` is not a mapping, iterable, or string URI.

### Examples

```pycon
>>> import pandas as pd
```

### Mapping case

```pycon
>>> tables_dict = {'table1': pd.DataFrame({'a': [1, 2], 'b': [3, 4]})}
>>> result = diagnose_table_collection(tables_dict)
>>> 'table1' in result
True
```

### Iterable case

```pycon
>>> df1 = pd.DataFrame({'a': [1, 2]})
>>> df2 = pd.DataFrame({'b': [3, 4]})
>>> result = diagnose_table_collection([df1, df2])
>>> 0 in result and 1 in result
True
```
