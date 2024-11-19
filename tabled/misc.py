"""
Miscellaneous utility functions for tables.
"""

from typing import List, Tuple
import pandas as pd


def scalar_columns(df: pd.DataFrame) -> List:
    """
    Returns the list of columns that are scalar (therefore serializable to CSV)

    More precisely, this function returns the list of columns that contain only
    scalar values (e.g., int, float, str, bool, etc.) and can be saved to a CSV
    file.

    Example:
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': ['x', 'y', 'z'],
    ...     'C': [{'a': 1}, {'b': 2}, {'c': 3}],  # Non-serializable column
    ...     'D': [[1, 2], [3, 4], [5, 6]]         # Non-serializable column
    ... })
    >>> scalar_columns(df)
    ['A', 'B']
    """
    import pandas.api.types as pdt

    return [col for col in df.columns if df[col].apply(pdt.is_scalar).all()]
