# tabled.util

General-purpose utilities for working with DataFrames, dicts, and byte decoding.

### Functions

| [`auto_decode_bytes`](#tabled.util.auto_decode_bytes)(b, \*[, try_first_bytes, ...])   | Decode a byte sequence into a string, trying charset_normalizer gueses if fails.                                                                             |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`breadth_first_traversal`](#tabled.util.breadth_first_traversal)(graph, start_node, \*)     | Yields nodes starting from the root node, expanding to neighbors recursively, using breadth-first search, without repeating nodes.                           |
| [`collapse_columns`](#tabled.util.collapse_columns)(df, groupings)                    | Transforms specified columns of a dataframe into single columns where each row contains a dictionary of column names and values from the original dataframe. |
| [`collapse_rows`](#tabled.util.collapse_rows)(df, by, \*[, container])             | Do a groupby to collapse (the rows of) a dataframe, gathering the other column's values (the ones that are not keys of the groupby) into lists.              |
| [`column_sep_key_mapper`](#tabled.util.column_sep_key_mapper)(key, column_name, sep)       | Join `column_name` and `key` with `sep` (e.g. `"X"`, `"a"`, `"."` -> `"X.a"`).                                                                               |
| [`duplicate_groups`](#tabled.util.duplicate_groups)(df, subset, \*[, output, ...])    | Get a DataFrame containing rows that have duplicate values for subset of columns.                                                                            |
| [`ensure_columns`](#tabled.util.ensure_columns)(df[, columns, fill])                | Ensure that a dataframe has certain columns, filling them with a certain value if they don't exist.                                                          |
| [`ensure_first_columns`](#tabled.util.ensure_first_columns)(df[, columns])                | Ensure that the given columns come first (if they exist), with the rest of the columns following in the order they were in the original dataframe.           |
| [`ensure_last_columns`](#tabled.util.ensure_last_columns)(df[, columns])                 | Ensure that the given columns come last (if they exist), with the rest of the columns preceding in the order they were in the original dataframe.            |
| [`expand_columns`](#tabled.util.expand_columns)(df, expand_columns, \*[, ...])      | Expands the iterable values of specified columns in to new columns.                                                                                          |
| [`expand_rows`](#tabled.util.expand_rows)(df, grouped_columns)                   | Expands a DataFrame where specific columns were collapsed into containers back to its original form.                                                         |
| [`identity`](#tabled.util.identity)(x)                                        | Return `x` unchanged.                                                                                                                                        |
| [`intersection_graph`](#tabled.util.intersection_graph)(sets[, edge_labels])            | A graph of all intersections between sets.                                                                                                                   |
| [`invert_labeled_collection`](#tabled.util.invert_labeled_collection)(d[, values_container])   | Invert a mapping whose values are iterables of objects, getting a mapping from objects to iterables of keys.                                                 |
| [`is_instance_of`](#tabled.util.is_instance_of)(class_or_tuple)                     | Return a predicate `obj -> isinstance(obj, class_or_tuple)`.                                                                                                 |
| [`is_non_null_or_empty`](#tabled.util.is_non_null_or_empty)(value)                        | Check if a value is not None, not empty, and not an empty list.                                                                                              |
| [`map_values`](#tabled.util.map_values)(func, d)                                | Apply a function to all values of a dictionary.                                                                                                              |
| [`split_keys`](#tabled.util.split_keys)(d)                                      | Returns a dictionary where keys that had spaces were split into multiple keys                                                                                |
| [`upsert_data`](#tabled.util.upsert_data)(target_df, source_df[, axis, ...])     | Updates or Inserts (Upserts) data into a target DataFrame, handling initial creation and growth along a specified axis.                                      |

### Classes

| [`PandasJSONEncoder`](#tabled.util.PandasJSONEncoder)(\*[, skipkeys, ...])   | A custom JSON encoder that can handle pandas and numpy types more robustly, even if they appear within nested data structures.   |
|-------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|

### *class* tabled.util.PandasJSONEncoder(, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, sort_keys=False, indent=None, separators=None, default=None)

Bases: [`JSONEncoder`](https://docs.python.org/3/library/json.html#json.JSONEncoder)

A custom JSON encoder that can handle pandas and numpy types more robustly,
even if they appear within nested data structures.

```pycon
>>> import json, datetime, pandas as pd, numpy as np
>>> # Test with a DataFrame containing timestamps and missing values.
>>> df = pd.DataFrame({
...     'a': [1, 2, 3],
...     'b': [pd.Timestamp('2023-04-09 00:02:53+0000', tz='UTC'),
...           pd.NaT,
...           pd.Timestamp('2023-04-09 00:02:53+0000', tz='UTC')]
... })
>>> json_str = json.dumps(df, cls=PandasJSONEncoder)
>>> json_str
'[{"a": 1, "b": "2023-04-09T00:02:53..."}..., {"a": 2, "b": null}, {"a": 3, "b": "2023-04-09T00:02:53..."}...]'
```

```pycon
>>> # Test with a Series containing timestamps and missing values.
>>> s = pd.Series([pd.Timestamp('2023-04-09 00:02:53+0000', tz='UTC'), pd.NaT])
>>> json_str = json.dumps(s, cls=PandasJSONEncoder)
>>> json_str
'{"0": "2023-04-09T00:02:53...", "1": null}'
```

```pycon
>>> # Test with numpy arrays and numpy scalar types.
>>> data = {
...     "arr": np.array([1, 2, 3], dtype=np.int32),
...     "flt": np.float32(3.14),
...     "bool": np.bool_(False)
... }
>>> json_str = json.dumps(data, cls=PandasJSONEncoder)
>>> json_str
'{"arr": [1, 2, 3], "flt": 3.14..., "bool": false}'
```

```pycon
>>> # Test with a datetime.date.
>>> date_val = datetime.date(2002, 1, 1)
>>> json.dumps(date_val, cls=PandasJSONEncoder)
'"2002-01-01"'
```

#### default(obj)

Convert `obj` (a pandas/numpy value the default encoder can’t handle) to a JSON-safe value.

### tabled.util.auto_decode_bytes(b, , try_first_bytes=(1000000.0, 10000000.0, 100000000.0), encoding='utf-8', verbose=False)

Decode a byte sequence into a string, trying charset_normalizer gueses if fails.

This function attempts to decode the given bytes using the default encoding (usually ‘utf-8’).
If that fails due to a `UnicodeDecodeError`, it uses `charset_normalizer` to detect the encoding
by analyzing increasingly larger samples of the byte sequence, as specified in `try_first_bytes`.
If all attempts fail, it analyzes the entire byte sequence to detect the encoding.

* **Parameters:**
  * **b** ([`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)) – The byte sequence to decode.
  * **try_first_bytes** – Byte lengths to use for encoding detection samples.
  * **encoding** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The encoding to try first, before falling back to detection.
  * **verbose** – If True, print each encoding tried.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
* **Returns:**
  The decoded string.
* **Raises:**
  [**UnicodeDecodeError**](https://docs.python.org/3/builtins/exceptions.html#UnicodeDecodeError) – If the byte sequence cannot be decoded after all attempts.

### Examples

```pycon
>>> # Example with UTF-8 encoded bytes
>>> s = 'Hello, world! Привет мир! こんにちは世界！'
>>> b_utf8 = s.encode('utf-8')
>>> auto_decode_bytes(b_utf8) == s
True
```

```pycon
>>> # Example with UTF-16 encoded bytes
>>> s_utf16 = 'Hello, world! 你好，世界！'
>>> b_utf16 = s_utf16.encode('utf-16')
>>> auto_decode_bytes(b_utf16) == s_utf16
True
```

Now, this is auto_decoding, but it doesn’t mean it’s robust.
We use `charset_normalizer` to detect the encoding of the bytes, and then
try to decode it with that encoding.
But sometimes you can decode something that is not the original string,
so be careful!!
It’s annoying to have to specify the encoding all the time, but this
explicitness, and the errors that come with it, can be vital.

Here are a few examples. We’ll

```pycon
>>> s_latin1 = 'Héllo, wörld! Ça va?'
>>> b_latin1 = s_latin1.encode('latin-1')  # latin-1 is ISO-8859-1
>>> decoded_s = auto_decode_bytes(b_latin1, verbose=True)
Trying encoding: 'utf-8'
Trying encoding: ...
>>> decoded_s
'H幨lo, w顤ld! ド va?'
>>> decoded_s == s_latin1
False
```

(Note in the above that some tests were skipped. This is because the output
is not deterministic and can vary depending on the system and the version of
`charset_normalizer`.)

```pycon
>>> s_cp1252 = 'Special characters: € £ ¥ © ®'
>>> b_cp1252 = s_cp1252.encode('cp1252')  # i.e. 'Windows-1252'
>>> decoded_s = auto_decode_bytes(b_cp1252, verbose=True)
Trying encoding: 'utf-8'
Trying encoding: 'cp1125'
>>> # See that charset_normalizer
>>> decoded_s
'Special characters: А г е й о'
```

### tabled.util.breadth_first_traversal(graph, start_node, , yield_edges=False)

Yields nodes starting from the root node, expanding to neighbors recursively,
using breadth-first search, without repeating nodes.

* **Parameters:**
  * **graph** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)) – Adjacencies of the graph: A mapping from nodes to their neighbors.
  * **start_node** – The node to start from (key of the graph adjacency mapping)
  * **yield_edges** – If True, yield edges instead of nodes.
    The edges are yielded as tuples of (node, neighbor).

```pycon
>>> graph = {
...     'A': ['B'], 'B': ['A', 'C', 'D'], 'C': ['B'], 'D': ['B', 'E'], 'E': ['D']
... }
>>> list(breadth_first_traversal(graph, 'B'))
['B', 'A', 'C', 'D', 'E']
>>> list(breadth_first_traversal(graph, 'B', yield_edges=True))
[('B', 'A'), ('B', 'C'), ('B', 'D'), ('D', 'E')]
```

### tabled.util.collapse_columns(df, groupings)

Transforms specified columns of a dataframe into single columns where each row
contains a dictionary of column names and values from the original dataframe.

* **Parameters:**
  * **df** (`DataFrame`) – The dataframe to transform.
  * **groupings** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]] | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)]) – A mapping that indicates which columns to collapse into dictionaries
    and what to call the new resulting column.
    If only a list of column names to be collapsed is given, it will be interpreted as
    the group_column_names in a single `{"collapsed": group_column_names}` dictionary,
    that is, all `group_column_names` are to be collapsed in to a single collapsed column.
* **Return type:**
  `DataFrame`
* **Returns:**
  A dataframe with the original columns not specified in `columns` untouched,
  and a new column `new_column_name` containing dictionaries of the collapsed columns.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If none of a grouping’s column names are found in `df`.

### Example

```pycon
>>> df = pd.DataFrame({
...     'a': [1, 1, 2, 2],
...     'b': [3, 4, 5, 6],
...     'c': [7, 8, 9, 10]
... })
>>> df
   a  b   c
0  1  3   7
1  1  4   8
2  2  5   9
3  2  6  10
>>> collapse_columns(df, {'ab': ['a', 'b']})
   c   ab
0  7  {'a': 1, 'b': 3}
1  8  {'a': 1, 'b': 4}
2  9  {'a': 2, 'b': 5}
3 10  {'a': 2, 'b': 6}
```

### tabled.util.collapse_rows(df, by, \*, container=<class 'list'>)

Do a groupby to collapse (the rows of) a dataframe, gathering the other
column’s values (the ones that are not keys of the groupby) into lists.

* **Parameters:**
  * **df** (`DataFrame`) – the dataframe to collapse
  * **by** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)]) – the columns to group by (the keys of the groupby)
  * **container** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)], [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)]) – the container to use to gather the other columns values
* **Return type:**
  `DataFrame`

```pycon
>>> df = pd.DataFrame({
...     'a': [1, 1, 2, 2],
...     'b': [3, 4, 5, 6],
...     'c': [7, 8, 9, 10]
... })
>>> df
   a  b   c
0  1  3   7
1  1  4   8
2  2  5   9
3  2  6  10
>>> collapse_rows(df, ['a'])
   a       b        c
0  1  [3, 4]   [7, 8]
1  2  [5, 6]  [9, 10]
```

### tabled.util.column_sep_key_mapper(key, column_name, sep)

Join `column_name` and `key` with `sep` (e.g. `"X"`, `"a"`, `"."` -> `"X.a"`).

### tabled.util.duplicate_groups(df, subset, , output='dataframe', keep_indices=True)

Get a DataFrame containing rows that have duplicate values for subset of columns.

* **Parameters:**
  * **df** – Input DataFrame
  * **subset** – Column name or list of column names to identify duplicates
  * **output** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Output format, either “dataframe” or “series”
  * **keep_indices** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True (default), preserves the original index as a column named
    by the index name or ‘index’ if unnamed. If False, keeps the
    original index as the index of the result.
* **Returns:**
  Series with unique duplicate values as index and corresponding DataFrames as values
  or DataFrame with duplicated rows with the specified subset as index.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If `output` is not `"dataframe"` or `"series"`.

```pycon
>>> import pandas as pd
>>> df = pd.DataFrame({"A": [1, 1, 2, 3, 3], "B": ["a", "b", "c", "d", "e"]})
>>> dups = duplicate_groups(df, "A")
>>> dups
   B  index
A
1  a      0
1  b      1
3  d      3
3  e      4
>>> list(dups.index)
[1, 1, 3, 3]
>>> dups.loc[1].shape
(2, 2)
>>> dups = duplicate_groups(df, "A", output="series")
>>> list(dups.index)
[1, 3]
>>> dups[1].shape
(2, 2)
>>> # Without keep_indices
>>> dups_orig_idx = duplicate_groups(df, "A", keep_indices=False)
>>> dups_orig_idx
   B
A
1  a
1  b
3  d
3  e
```

### tabled.util.ensure_columns(df, columns=(), fill=None)

Ensure that a dataframe has certain columns, filling them with a certain value
if they don’t exist.

### tabled.util.ensure_first_columns(df, columns=())

Ensure that the given columns come first (if they exist), with the rest of the columns
following in the order they were in the original dataframe.

### tabled.util.ensure_last_columns(df, columns=())

Ensure that the given columns come last (if they exist), with the rest of the columns
preceding in the order they were in the original dataframe.

### tabled.util.expand_columns(df, expand_columns, \*, drop=True, key_mapper=functools.partial(<function column_sep_key_mapper>, sep='.'), drop_non_iterable_rows=False)

Expands the iterable values of specified columns in to new columns.
The new columns will be named using the column_name and the key of the values of
the iterable that is expanded (key if dict, integer index if sequence).

* **Parameters:**
  * **df** (`DataFrame`) – The dataframe to transform.
  * **expand_columns** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – A list of column names whose values are dictionaries
    that need to be expanded into new columns.
  * **drop** – Whether to drop the original columns that were expanded.
  * **drop_non_iterable_rows** – Whether to drop rows that have non-iterable values
  * **key_mapper** – A function that takes a key and a column name and returns a
    new key. By default, the new key is the concatenation of the column name and
    the original key. If None, will just take the original key. The reason for
    also taking the column_name by default is to avoid collisions if the keys are
    used in more than one column.
* **Return type:**
  `DataFrame`
* **Returns:**
  A dataframe with the expanded columns added.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If a name in `expand_columns` is not a column of `df`.

### Examples

```pycon
>>> df = pd.DataFrame({
...     'c': [7, 8, 9, 10],
...     'X': [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 5}, {'a': 2, 'b': 6}]
... })
>>> expand_columns(df, ['X'])
   c  X.a  X.b
0  7  1  3
1  8  1  4
2  9  2  5
3 10  2  6
```

Let’s see what happens when the elements of an expanded column are lists instead of
dicts, we ask to not drop, and we use `key_mapper=None`:

```pycon
>>> df = pd.DataFrame({
...     'c': [7, 8, 9, 10],
...     'X': [[1, 3], [1, 4], [2, 5], [2, 6]]
... })
>>> expand_columns(df, ['X'], drop=False, key_mapper=None)
    c       X  0  1
0   7  [1, 3]  1  3
1   8  [1, 4]  1  4
2   9  [2, 5]  2  5
3  10  [2, 6]  2  6
```

### tabled.util.expand_rows(df, grouped_columns)

Expands a DataFrame where specific columns were collapsed into containers back to its original form.
Each column in `grouped_columns` should contain lists of the same length within each row.

* **Parameters:**
  * **df** (`DataFrame`) – The DataFrame to expand.
  * **grouped_columns** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)]) – The list of columns to expand
* **Return type:**
  `DataFrame`
* **Returns:**
  The expanded DataFrame.

```pycon
>>> df_collapsed = pd.DataFrame({
...     'a': [1, 2],
...     'b': [[3, 4], [5, 6, 66]],
...     'c': [[7, 8], [9, 10, 11]]
... })
>>> expand_rows(df_collapsed, ['b', 'c'])
    a  b   c
0  1  3   7
1  1  4   8
2  2  5   9
3  2  6  10
4  2  66  11
```

### tabled.util.identity(x)

Return `x` unchanged.

### tabled.util.intersection_graph(sets, edge_labels=False)

A graph of all intersections between sets.
(See [https://en.wikipedia.org/wiki/Intersection_graph](https://en.wikipedia.org/wiki/Intersection_graph).)

In graph theory, an adjacency list is a collection of sets used to represent a
finite graph.
Here, the vertices are the values of sets,
and there is an edge between two vertices if the sets intersect.
The weight of the edge is the size of the intersection.

* **Parameters:**
  * **sets** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`set`](https://docs.python.org/3/builtins/stdtypes.html#set)]) – A mapping of keys to sets of elements. These sets of elements will
    be the vertices of the graph.
  * **edge_labels** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'elements'`, `'size'`, `False`]) – If ‘elements’, the edge labels are the elements of the intersection.
    If ‘size’, the edge labels are the size of the intersection.
    If False, there are no edge labels.
* **Returns:**
  A graph, represented by an “adjacency list”
  (see [https://en.wikipedia.org/wiki/Adjacency_list](https://en.wikipedia.org/wiki/Adjacency_list))
  (a dict whose keys are the keys of the input `sets` dict, and whose values
  tell us what sets of `sets` intersect with it),
  optionally with some information about this intersection.

```pycon
>>> sets = {
...     'A': {'b', 'c'},
...     'B': {'a', 'b', 'd', 'e', 'f'},
...     'C': {'f', 'g'},
...     'D': {'d', 'e', 'h', 'i'},
...     'E': {'i', 'j'}
... }
>>> assert intersection_graph(sets) == {
...     'A': {'B'}, 'B': {'A', 'C', 'D'}, 'C': {'B'}, 'D': {'B', 'E'}, 'E': {'D'}
... }
>>> assert intersection_graph(sets, edge_labels='elements') == {
...     'A': {'B': {'b'}},
...     'B': {'A': {'b'}, 'C': {'f'}, 'D': {'d', 'e'}},
...     'C': {'B': {'f'}},
...     'D': {'B': {'d', 'e'}, 'E': {'i'}},
...     'E': {'D': {'i'}}
... }
>>> assert intersection_graph(sets, edge_labels='size') == {
...     'A': {'B': 1},
...     'B': {'A': 1, 'C': 1, 'D': 2},
...     'C': {'B': 1},
...     'D': {'B': 2, 'E': 1},
...     'E': {'D': 1}
... }
```

### tabled.util.invert_labeled_collection(d, values_container=<class 'list'>)

Invert a mapping whose values are iterables of objects,
getting a mapping from objects to iterables of keys.

* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`VT`), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)]]

```pycon
>>> original_dict = {
...     "X": ['a', 'b'],
...     "Y": ['a'],
...     "Z": ['a', 'b', 'c']
... }
>>> inverted_dict = invert_labeled_collection(original_dict)
>>> inverted_dict
{'a': ['X', 'Y', 'Z'], 'b': ['X', 'Z'], 'c': ['Z']}
>>> invert_labeled_collection(inverted_dict)
{'X': ['a', 'b'], 'Y': ['a'], 'Z': ['a', 'b', 'c']}
```

The `values_container` argument can be used to cast the values of the inverted dict.

```pycon
>>> assert (
...     invert_labeled_collection(original_dict, values_container=set)
...     == {'a': {'X', 'Y', 'Z'}, 'b': {'X', 'Z'}, 'c': {'Z'}}
... )
>>>
>>> d = {'a': 'apple', 'b': 'banana'}
>>> t = invert_labeled_collection(d, values_container=''.join)
>>> t
{'a': 'abbb', 'p': 'aa', 'l': 'a', 'e': 'a', 'b': 'b', 'n': 'bb'}
>>> invert_labeled_collection(t, ''.join)
{'a': 'apple', 'b': 'aaabnn'}
```

### tabled.util.is_instance_of(class_or_tuple)

Return a predicate `obj -> isinstance(obj, class_or_tuple)`.

### tabled.util.is_non_null_or_empty(value)

Check if a value is not None, not empty, and not an empty list.

Often used with pandas dataframes to check if a cell is null or non-empty.

```text
num_of_non_empties_in_row = df.map(is_non_null_or_empty).sum(axis=1)
num_of_non_empties_in_col = df.map(is_non_null_or_empty).sum(axis=0)
```

And then you can do:

```text
num_of_non_empties_in_row.sort_values(ascending=False) to see which rows have the least empties (most actual data)
```

### tabled.util.map_values(func, d)

Apply a function to all values of a dictionary.

```pycon
>>> map_values(lambda x: x ** 2, {1: 2, 3: 4})
{1: 4, 3: 16}
```

### tabled.util.split_keys(d)

Returns a dictionary where keys that had spaces were split into multiple keys

Meant to be a convenience function for the user to use when they want to define a
mapping where several keys map to the same value.

```pycon
>>> split_keys({'apple': 1, 'banana carrot': 2})
{'apple': 1, 'banana': 2, 'carrot': 2}
```

### tabled.util.upsert_data(target_df, source_df, axis=1, align_index_value=False)

Updates or Inserts (Upserts) data into a target DataFrame, handling initial
creation and growth along a specified axis.

The function returns a new DataFrame, even though it modifies the target
in-place for column updates (axis=1).

* **Parameters:**
  * **target_df** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[`DataFrame`]) – The DataFrame to be updated (can be None or empty).
  * **source_df** (`DataFrame`) – The DataFrame containing the new/source data.
  * **axis** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`0`, `1`]) – 1 for adding columns (default), 0 for adding rows.
  * **align_index_value** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, the concatenation requires indices (or columns
    when axis=0) to match both in number AND value.
    If False, alignment is ignored (positional concat).
* **Return type:**
  `DataFrame`
* **Returns:**
  The updated DataFrame.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If `align_index_value` is `False` and the row count (axis=1)
      or column count (axis=0) of `target_df` and `source_df` don’t match.

### Examples

```pycon
>>> # 1. Initial creation (target_df is None)
>>> df_target = None
>>> df_source_A = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
>>> df_target = upsert_data(df_target, df_source_A)
>>> df_target.equals(df_source_A)
True
```

```pycon
>>> # 2. Adding new columns (axis=1, default)
>>> df_source_B = pd.DataFrame({'c': [5, 6], 'd': [7, 8]})
>>> df_target = upsert_data(df_target, df_source_B)
>>> df_target.columns.tolist()
['a', 'b', 'c', 'd']
```

```pycon
>>> # 3. Overwriting existing columns (axis=1)
>>> # Indices must align when updating
>>> df_source_C = pd.DataFrame({'a': [10, 20], 'e': [30, 40]})
>>> df_target = upsert_data(df_target, df_source_C)
>>> df_target['a'].tolist()
[10, 20]
>>> df_target.columns.tolist()
['a', 'b', 'c', 'd', 'e']
```

```pycon
>>> # 4. Adding rows (axis=0)
>>> df_target_row = pd.DataFrame({'col1': [1], 'col2': [2]})
>>> df_source_row = pd.DataFrame({'col1': [3], 'col2': [4]})
>>> df_target_row = upsert_data(df_target_row, df_source_row, axis=0)
>>> df_target_row.shape
(2, 2)
>>> df_target_row['col1'].tolist()
[1, 3]
```
