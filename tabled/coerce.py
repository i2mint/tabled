"""
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

Example
-------
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
"""

from typing import Callable, Optional, List
import json


def coerce_series_conditionally(
    series: "pd.Series",
    condition: Callable,
    transform: Callable,
    *,
    sample_size: int = 100,
    threshold: float = 0.8,
) -> "pd.Series":
    """
    Conditionally transform a pandas Series based on sampled values.

    This function samples non-null values to check if a condition holds,
    and if so, applies the transform to all non-null values. This is useful
    for efficiently detecting and converting columns with serialized data.

    Parameters
    ----------
    series : pd.Series
        The series to potentially transform.
    condition : callable
        A function (value -> bool) that tests whether a value needs
        transformation. Applied to a sample to determine if transformation
        should occur for the whole series.
    transform : callable
        A function (value -> new_value) to apply to each non-null value
        if the condition threshold is met.
    sample_size : int
        Maximum number of non-null values to sample for condition testing.
        Larger samples give more reliable detection but cost more time.
    threshold : float
        Fraction of sampled values that must satisfy the condition (0.0 to 1.0)
        for the transform to be applied. Use lower values for columns with
        mixed or partially valid data.

    Returns
    -------
    pd.Series
        The original series if condition not met, otherwise a new series
        with transformed values (null values are preserved).

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series(['[1, 2]', '[3, 4]', None, '[5]'])
    >>> is_json_list = lambda x: isinstance(x, str) and x.startswith('[')
    >>> import json
    >>> result = coerce_series_conditionally(s, is_json_list, json.loads)
    >>> result.iloc[0]
    [1, 2]
    >>> pd.isna(result.iloc[2])  # null values are preserved (None or NaN)
    True

    Notes
    -----
    The function uses a fixed random_state (42) for reproducible sampling.
    If the series has fewer non-null values than sample_size, all non-null
    values are used for testing.
    """
    import pandas as pd

    # Get non-null values
    non_null_mask = series.notna()
    non_null = series[non_null_mask]

    if len(non_null) == 0:
        return series

    # Sample for condition testing
    actual_sample_size = min(sample_size, len(non_null))
    sample = non_null.sample(n=actual_sample_size, random_state=42)

    # Check if condition holds for enough sampled values
    condition_met = sample.apply(condition).sum() / len(sample) >= threshold

    if not condition_met:
        return series

    # Apply transform to non-null values only
    transformed = non_null.apply(transform)

    # If the transformed values have a different type (e.g., strings -> lists),
    # we need to convert to object dtype to avoid Arrow string array errors
    result = series.copy()
    if result.dtype != "object":
        result = result.astype("object")

    result.loc[non_null_mask] = transformed
    return result


def is_json_string(value) -> bool:
    """
    Check if a value looks like a JSON-encoded string.

    Detects strings that appear to contain JSON arrays or objects
    (starting with '[' or '{' and ending with ']' or '}').

    Parameters
    ----------
    value : any
        The value to check.

    Returns
    -------
    bool
        True if the value appears to be a JSON string.

    Examples
    --------
    >>> is_json_string('[1, 2, 3]')
    True
    >>> is_json_string('{"key": "value"}')
    True
    >>> is_json_string('plain text')
    False
    >>> is_json_string(123)
    False
    """
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return (stripped.startswith("[") and stripped.endswith("]")) or (
        stripped.startswith("{") and stripped.endswith("}")
    )


def is_json_list_string(value) -> bool:
    """
    Check if a value looks like a JSON list encoded as a string.

    Parameters
    ----------
    value : any
        The value to check.

    Returns
    -------
    bool
        True if the value appears to be a JSON list string.

    Examples
    --------
    >>> is_json_list_string('[1, 2, 3]')
    True
    >>> is_json_list_string('["a", "b"]')
    True
    >>> is_json_list_string('{"key": "value"}')
    False
    >>> is_json_list_string('not json')
    False
    """
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return stripped.startswith("[") and stripped.endswith("]")


def is_json_dict_string(value) -> bool:
    """
    Check if a value looks like a JSON object/dict encoded as a string.

    Parameters
    ----------
    value : any
        The value to check.

    Returns
    -------
    bool
        True if the value appears to be a JSON object string.

    Examples
    --------
    >>> is_json_dict_string('{"key": "value"}')
    True
    >>> is_json_dict_string('[1, 2, 3]')
    False
    """
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return stripped.startswith("{") and stripped.endswith("}")


def parse_json_safe(value):
    """
    Parse a JSON string, returning the original value if parsing fails.

    This is a safe wrapper around json.loads that never raises exceptions,
    making it suitable for use with coerce_series_conditionally on columns
    that may contain some malformed JSON.

    Parameters
    ----------
    value : any
        The value to parse. If not a valid JSON string, returned as-is.

    Returns
    -------
    any
        The parsed JSON value, or the original value if parsing failed.

    Examples
    --------
    >>> parse_json_safe('[1, 2, 3]')
    [1, 2, 3]
    >>> parse_json_safe('{"a": 1}')
    {'a': 1}
    >>> parse_json_safe('not json')
    'not json'
    >>> parse_json_safe(None)  # Non-strings pass through
    """
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def coerce_json_column(series: "pd.Series", **kwargs) -> "pd.Series":
    """
    Coerce a column of JSON strings to Python objects.

    Detects and converts strings that contain JSON arrays or objects
    to their Python equivalents (lists or dicts).

    Parameters
    ----------
    series : pd.Series
        The series to coerce.
    **kwargs
        Additional arguments passed to coerce_series_conditionally
        (sample_size, threshold).

    Returns
    -------
    pd.Series
        Series with JSON strings converted to Python objects.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series(['[1, 2]', '{"a": 1}', 'text', None])
    >>> # Note: won't transform if < 80% are JSON by default
    >>> s_homogeneous = pd.Series(['[1]', '[2]', '[3]', None])
    >>> coerced = coerce_json_column(s_homogeneous)
    >>> coerced.iloc[0]
    [1]
    """
    return coerce_series_conditionally(
        series,
        condition=is_json_string,
        transform=parse_json_safe,
        **kwargs,
    )


def coerce_json_list_column(series: "pd.Series", **kwargs) -> "pd.Series":
    """
    Coerce a column of JSON list strings to Python lists.

    This is a convenience wrapper around coerce_series_conditionally
    configured for the common case of columns containing JSON list strings.

    Parameters
    ----------
    series : pd.Series
        The series to coerce.
    **kwargs
        Additional arguments passed to coerce_series_conditionally
        (sample_size, threshold).

    Returns
    -------
    pd.Series
        Series with JSON list strings converted to Python lists.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series(['[1, 2, 3]', '["a", "b"]', None])
    >>> coerced = coerce_json_list_column(s)
    >>> coerced.iloc[0]
    [1, 2, 3]
    >>> coerced.iloc[1]
    ['a', 'b']
    """
    return coerce_series_conditionally(
        series,
        condition=is_json_list_string,
        transform=parse_json_safe,
        **kwargs,
    )


def coerce_dataframe_columns(
    df: "pd.DataFrame",
    condition: Callable,
    transform: Callable,
    columns: Optional[List[str]] = None,
    *,
    sample_size: int = 100,
    threshold: float = 0.8,
    verbose: bool = False,
) -> "pd.DataFrame":
    """
    Conditionally coerce columns in a DataFrame.

    Applies coerce_series_conditionally to specified columns (or all
    object-dtype columns if none specified).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.
    condition : callable
        A function (value -> bool) to test if values need transformation.
    transform : callable
        A function (value -> new_value) to apply to matching values.
    columns : list of str or None
        Specific columns to check. If None, checks all object-dtype columns.
    sample_size : int
        Maximum number of values to sample per column.
    threshold : float
        Fraction of samples that must satisfy condition to trigger transform.
    verbose : bool
        If True, print which columns were transformed.

    Returns
    -------
    pd.DataFrame
        DataFrame with coerced columns.

    Examples
    --------
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
    """
    df = df.copy()

    if columns is None:
        # Only check object columns (strings)
        columns = df.select_dtypes(include=["object"]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        original = df[col]
        coerced = coerce_series_conditionally(
            original,
            condition=condition,
            transform=transform,
            sample_size=sample_size,
            threshold=threshold,
        )

        # Check if any transformation occurred
        if not original.equals(coerced):
            df[col] = coerced
            if verbose:
                print(f"  Coerced column '{col}'")

    return df


def coerce_json_columns(
    df: "pd.DataFrame",
    columns: Optional[List[str]] = None,
    *,
    verbose: bool = False,
    **kwargs,
) -> "pd.DataFrame":
    """
    Coerce JSON string columns in a DataFrame to Python objects.

    A convenience function that applies JSON coercion to specified columns
    (or all object-dtype columns) in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.
    columns : list of str or None
        Specific columns to check. If None, checks all object-dtype columns.
    verbose : bool
        If True, print which columns were transformed.
    **kwargs
        Additional arguments passed to coerce_series_conditionally
        (sample_size, threshold).

    Returns
    -------
    pd.DataFrame
        DataFrame with JSON string columns coerced to Python objects.

    Examples
    --------
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
    """
    return coerce_dataframe_columns(
        df,
        condition=is_json_string,
        transform=parse_json_safe,
        columns=columns,
        verbose=verbose,
        **kwargs,
    )
