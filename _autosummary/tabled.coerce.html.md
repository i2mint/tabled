# tabled.coerce

Column type coercion utilities for pandas DataFrames.

This module provides tools for conditionally transforming DataFrame columns
based on sampled value inspection. The primary use case is detecting and
converting columns that contain serialized data (e.g., JSON strings that
should be lists or dicts).

The design follows a sample-then-transform pattern:

1. Sample a subset of non-null values from a column
2. Test if a condition holds for a threshold fraction of samples
3. If so, apply a transformation to all non-null values

This approach is efficient for large DataFrames where checking every value
would be expensive, and robust to mixed or partially malformed data.

### Example

```pycon
>>> import pandas as pd
>>> from tabled.coerce import coerce_json_list_column
>>>
>>> # A column with JSON list strings
>>> s = pd.Series(['[1, 2, 3]', '["a", "b"]', None, '[4, 5]'])
>>> coerced = coerce_json_list_column(s)
>>> coerced.iloc[0]
[1, 2, 3]
>>> coerced.iloc[1]
['a', 'b']
```

### Functions

| [`coerce_dataframe_columns`](#tabled.coerce.coerce_dataframe_columns)(df, condition, ...)    | Conditionally coerce columns in a DataFrame.                        |
|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| [`coerce_json_column`](#tabled.coerce.coerce_json_column)(series, \*\*kwargs)          | Coerce a column of JSON strings to Python objects.                  |
| [`coerce_json_columns`](#tabled.coerce.coerce_json_columns)(df[, columns, verbose])     | Coerce JSON string columns in a DataFrame to Python objects.        |
| [`coerce_json_list_column`](#tabled.coerce.coerce_json_list_column)(series, \*\*kwargs)     | Coerce a column of JSON list strings to Python lists.               |
| [`coerce_series_conditionally`](#tabled.coerce.coerce_series_conditionally)(series, ...[, ...]) | Conditionally transform a pandas Series based on sampled values.    |
| [`is_json_dict_string`](#tabled.coerce.is_json_dict_string)(value)                      | Check if a value looks like a JSON object/dict encoded as a string. |
| [`is_json_list_string`](#tabled.coerce.is_json_list_string)(value)                      | Check if a value looks like a JSON list encoded as a string.        |
| [`is_json_string`](#tabled.coerce.is_json_string)(value)                           | Check if a value looks like a JSON-encoded string.                  |
| [`parse_json_safe`](#tabled.coerce.parse_json_safe)(value)                          | Parse a JSON string, returning the original value if parsing fails. |

### tabled.coerce.coerce_dataframe_columns(df, condition, transform, columns=None, , sample_size=100, threshold=0.8, verbose=False)

Conditionally coerce columns in a DataFrame.

Applies coerce_series_conditionally to specified columns (or all
object-dtype columns if none specified).

* **Parameters:**
  * **df** (`DataFrame`) – The DataFrame to process.
  * **condition** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – A function (value -> bool) to test if values need transformation.
  * **transform** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – A function (value -> new_value) to apply to matching values.
  * **columns** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Specific columns to check. If None, checks all object-dtype columns.
  * **sample_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number of values to sample per column.
  * **threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Fraction of samples that must satisfy condition to trigger transform.
  * **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, print which columns were transformed.
* **Returns:**
  DataFrame with coerced columns.
* **Return type:**
  `DataFrame`

### Examples

```pycon
>>> import pandas as pd
>>> df = pd.DataFrame({
...     'json_col': ['[1]', '[2]', '[3]'],
...     'text_col': ['a', 'b', 'c']
... })
>>> from tabled.coerce import is_json_list_string, parse_json_safe
>>> result = coerce_dataframe_columns(
...     df, is_json_list_string, parse_json_safe
... )
>>> result['json_col'].iloc[0]
[1]
>>> result['text_col'].iloc[0]  # unchanged - not a JSON list
'a'
```

### tabled.coerce.coerce_json_column(series, \*\*kwargs)

Coerce a column of JSON strings to Python objects.

Detects and converts strings that contain JSON arrays or objects
to their Python equivalents (lists or dicts).

* **Parameters:**
  * **series** (`Series`) – The series to coerce.
  * **\*\*kwargs** – Additional arguments passed to coerce_series_conditionally
    (sample_size, threshold).
* **Returns:**
  Series with JSON strings converted to Python objects.
* **Return type:**
  `Series`

### Examples

```pycon
>>> import pandas as pd
>>> s = pd.Series(['[1, 2]', '{"a": 1}', 'text', None])
>>> # Note: won't transform if < 80% are JSON by default
>>> s_homogeneous = pd.Series(['[1]', '[2]', '[3]', None])
>>> coerced = coerce_json_column(s_homogeneous)
>>> coerced.iloc[0]
[1]
```

### tabled.coerce.coerce_json_columns(df, columns=None, , verbose=False, \*\*kwargs)

Coerce JSON string columns in a DataFrame to Python objects.

A convenience function that applies JSON coercion to specified columns
(or all object-dtype columns) in a DataFrame.

* **Parameters:**
  * **df** (`DataFrame`) – The DataFrame to process.
  * **columns** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`List`](https://docs.python.org/3/library/typing.html#typing.List)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Specific columns to check. If None, checks all object-dtype columns.
  * **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, print which columns were transformed.
  * **\*\*kwargs** – Additional arguments passed to coerce_series_conditionally
    (sample_size, threshold).
* **Returns:**
  DataFrame with JSON string columns coerced to Python objects.
* **Return type:**
  `DataFrame`

### Examples

```pycon
>>> import pandas as pd
>>> df = pd.DataFrame({
...     'tags': ['["a", "b"]', '["c"]', '["d", "e", "f"]'],
...     'ids': ['[1, 2]', '[3]', '[4, 5]'],
...     'name': ['Alice', 'Bob', 'Charlie']
... })
>>> result = coerce_json_columns(df)
>>> result['tags'].iloc[0]
['a', 'b']
>>> result['name'].iloc[0]  # Unchanged - not JSON
'Alice'
```

### tabled.coerce.coerce_json_list_column(series, \*\*kwargs)

Coerce a column of JSON list strings to Python lists.

This is a convenience wrapper around coerce_series_conditionally
configured for the common case of columns containing JSON list strings.

* **Parameters:**
  * **series** (`Series`) – The series to coerce.
  * **\*\*kwargs** – Additional arguments passed to coerce_series_conditionally
    (sample_size, threshold).
* **Returns:**
  Series with JSON list strings converted to Python lists.
* **Return type:**
  `Series`

### Examples

```pycon
>>> import pandas as pd
>>> s = pd.Series(['[1, 2, 3]', '["a", "b"]', None])
>>> coerced = coerce_json_list_column(s)
>>> coerced.iloc[0]
[1, 2, 3]
>>> coerced.iloc[1]
['a', 'b']
```

### tabled.coerce.coerce_series_conditionally(series, condition, transform, , sample_size=100, threshold=0.8)

Conditionally transform a pandas Series based on sampled values.

This function samples non-null values to check if a condition holds,
and if so, applies the transform to all non-null values. This is useful
for efficiently detecting and converting columns with serialized data.

* **Parameters:**
  * **series** (`Series`) – The series to potentially transform.
  * **condition** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – A function (value -> bool) that tests whether a value needs
    transformation. Applied to a sample to determine if transformation
    should occur for the whole series.
  * **transform** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)) – A function (value -> new_value) to apply to each non-null value
    if the condition threshold is met.
  * **sample_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number of non-null values to sample for condition testing.
    Larger samples give more reliable detection but cost more time.
  * **threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Fraction of sampled values that must satisfy the condition (0.0 to 1.0)
    for the transform to be applied. Use lower values for columns with
    mixed or partially valid data.
* **Returns:**
  The original series if condition not met, otherwise a new series
  with transformed values (null values are preserved).
* **Return type:**
  `Series`

### Examples

```pycon
>>> import pandas as pd
>>> s = pd.Series(['[1, 2]', '[3, 4]', None, '[5]'])
>>> is_json_list = lambda x: isinstance(x, str) and x.startswith('[')
>>> import json
>>> result = coerce_series_conditionally(s, is_json_list, json.loads)
>>> result.iloc[0]
[1, 2]
>>> pd.isna(result.iloc[2])  # null values are preserved (None or NaN)
True
```

### Notes

The function uses a fixed random_state (42) for reproducible sampling.
If the series has fewer non-null values than sample_size, all non-null
values are used for testing.

### tabled.coerce.is_json_dict_string(value)

Check if a value looks like a JSON object/dict encoded as a string.

* **Parameters:**
  **value** (*any*) – The value to check.
* **Returns:**
  True if the value appears to be a JSON object string.
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### Examples

```pycon
>>> is_json_dict_string('{"key": "value"}')
True
>>> is_json_dict_string('[1, 2, 3]')
False
```

### tabled.coerce.is_json_list_string(value)

Check if a value looks like a JSON list encoded as a string.

* **Parameters:**
  **value** (*any*) – The value to check.
* **Returns:**
  True if the value appears to be a JSON list string.
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### Examples

```pycon
>>> is_json_list_string('[1, 2, 3]')
True
>>> is_json_list_string('["a", "b"]')
True
>>> is_json_list_string('{"key": "value"}')
False
>>> is_json_list_string('not json')
False
```

### tabled.coerce.is_json_string(value)

Check if a value looks like a JSON-encoded string.

Detects strings that appear to contain JSON arrays or objects
(starting with ‘[’ or ‘{’ and ending with ‘]’ or ‘}’).

* **Parameters:**
  **value** (*any*) – The value to check.
* **Returns:**
  True if the value appears to be a JSON string.
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### Examples

```pycon
>>> is_json_string('[1, 2, 3]')
True
>>> is_json_string('{"key": "value"}')
True
>>> is_json_string('plain text')
False
>>> is_json_string(123)
False
```

### tabled.coerce.parse_json_safe(value)

Parse a JSON string, returning the original value if parsing fails.

This is a safe wrapper around json.loads that never raises exceptions,
making it suitable for use with coerce_series_conditionally on columns
that may contain some malformed JSON.

* **Parameters:**
  **value** (*any*) – The value to parse. If not a valid JSON string, returned as-is.
* **Returns:**
  The parsed JSON value, or the original value if parsing failed.
* **Return type:**
  any

### Examples

```pycon
>>> parse_json_safe('[1, 2, 3]')
[1, 2, 3]
>>> parse_json_safe('{"a": 1}')
{'a': 1}
>>> parse_json_safe('not json')
'not json'
>>> parse_json_safe(None)  # Non-strings pass through
```
