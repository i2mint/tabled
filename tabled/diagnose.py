"""
DataFrame and table collection diagnosis utilities.

This module provides flexible tools for analyzing pandas DataFrames and collections
of tables. The core function `dataframe_info` extracts configurable information
from DataFrames using pluggable info functions.

Key Features:
- Configurable info extraction with `dataframe_info`
- Collection diagnosis with `diagnose_table_collection`
- Extensible via custom info functions
- Backward-compatible `print_dataframe_info` from cosmodata

Example:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    >>> info = dataframe_info(df)
    >>> info['shape']
    (3, 2)

    # Register custom info function
    >>> def get_memory_usage(df):
    ...     return df.memory_usage(deep=True).sum()
    >>> register_info_func('custom_memory', get_memory_usage)
"""

from typing import List, Tuple
import pandas as pd


def scalar_columns(df: pd.DataFrame) -> list:
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


# --------------------------------------------------------------------------------------

from typing import Literal, Callable, Optional, Union
import pandas as pd


# Default info functions for DataFrame analysis
def _get_shape(df: pd.DataFrame) -> tuple[int, int]:
    """Get DataFrame shape."""
    return df.shape


def _get_columns(df: pd.DataFrame) -> list[str]:
    """Get DataFrame column names."""
    return list(df.columns)


def _get_first_row(df: pd.DataFrame) -> pd.Series:
    """Get first row of DataFrame."""
    return df.iloc[0] if len(df) > 0 else pd.Series(dtype=object)


def _get_sample_rows(df: pd.DataFrame, n_samples: int = 3) -> pd.DataFrame:
    """Get random sample of DataFrame rows."""
    n_samples = min(n_samples, len(df))
    return df.sample(n=n_samples) if len(df) > 0 else pd.DataFrame()


def _get_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Get descriptive statistics for numeric columns."""
    numeric_cols = df.select_dtypes(include="number").columns
    return df[numeric_cols].describe() if len(numeric_cols) > 0 else pd.DataFrame()


def _get_categorical_stats(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Get value counts for categorical columns.

    Only includes columns with hashable (scalar) values. Columns containing
    arrays, lists, dicts, or other unhashable types are skipped.
    """
    # Get non-numeric columns
    categorical_cols = df.select_dtypes(exclude="number").columns

    # Filter to only scalar (hashable) columns to avoid errors with arrays/lists
    scalar_cols = set(scalar_columns(df))
    categorical_cols = [col for col in categorical_cols if col in scalar_cols]

    result = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts().head(5)
        result[col] = {"value_counts": value_counts, "total_unique": df[col].nunique()}
    return result


# Default info functions dictionary
DFLT_INFO_FUNCS = {
    "shape": _get_shape,
    "columns": _get_columns,
    "first_row": _get_first_row,
    "sample_rows": _get_sample_rows,
    "numeric_stats": _get_numeric_stats,
    "categorical_stats": _get_categorical_stats,
}


def register_info_func(
    name: str, func: Callable[[pd.DataFrame], any], *, overwrite: bool = False
):
    """
    Register a new info function in the default info functions dictionary.

    Args:
        name: Name for the info function
        func: Function that takes a DataFrame and returns info
        overwrite: Whether to overwrite existing functions with the same name

    Example:
        >>> def get_memory_usage(df):
        ...     return df.memory_usage(deep=True).sum()
        >>> register_info_func('test_memory', get_memory_usage)
    """
    if name in DFLT_INFO_FUNCS and not overwrite:
        raise ValueError(
            f"Info function '{name}' already exists. Use overwrite=True to replace."
        )
    DFLT_INFO_FUNCS[name] = func


def list_info_funcs() -> list[str]:
    """List all registered info function names."""
    return list(DFLT_INFO_FUNCS.keys())


# Mode-specific info function selections
_MODE_INFO_FUNCS = {
    "short": {
        "shape": _get_shape,
        "first_row": _get_first_row,
    },
    "sample": {
        "shape": _get_shape,
        "columns": _get_columns,
        "sample_rows": _get_sample_rows,
    },
    "stats": {
        "shape": _get_shape,
        "numeric_stats": _get_numeric_stats,
        "categorical_stats": _get_categorical_stats,
    },
}


def dataframe_info(
    df: pd.DataFrame,
    info_funcs: dict[str, Callable] = DFLT_INFO_FUNCS,
    *,
    egress: Callable = dict,
):
    """Extract information from a DataFrame using specified info functions.

    Args:
        df: The DataFrame to analyze
        info_funcs: Dict mapping info keys to functions that take a DataFrame
        egress: Function to process the generator of (key, value) pairs

    Returns:
        Result of egress applied to the info generator

    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> info = dataframe_info(df, {'shape': _get_shape})
    >>> info['shape']
    (3, 2)
    """

    def info_gen():
        for info_key, info_func in info_funcs.items():
            yield info_key, info_func(df)

    return egress(info_gen())


def print_dataframe_info(
    df: pd.DataFrame,
    exclude_columns: Union[str, list[str]] = (),
    *,
    mode: Literal["short", "sample", "stats"] = "short",
    egress: Optional[Callable[[str], None]] = print,
):
    """Print information about a DataFrame.

    Args:
        df: The DataFrame to analyze
        exclude_columns: Columns to exclude from analysis
        mode: Type of information to display
            - 'short': shape and first row
            - 'sample': shape, columns, and random rows
            - 'stats': descriptive statistics
        egress: Callback function for output (None returns string instead of printing)

    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> info = print_dataframe_info(df, egress=None)
    >>> 'shape: (3, 2)' in info
    True
    """
    if isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]
    if egress == "copy":
        import pyperclip  # pip install pyperclip

        egress = pyperclip.copy

    if exclude_columns:
        df = df.drop(columns=exclude_columns, errors="ignore")

    # Get the appropriate info functions for the mode
    info_funcs = _MODE_INFO_FUNCS.get(mode)
    if info_funcs is None:
        raise ValueError(f"Unknown mode: {mode}. Use 'short', 'sample', or 'stats'.")

    # Get the information
    info_dict = dataframe_info(df, info_funcs)

    # Format the information as a string
    s = _format_dataframe_info(info_dict, mode)

    if not egress:
        return s
    return egress(s)


def _format_dataframe_info(info_dict: dict, mode: str) -> str:
    """Format the DataFrame info dictionary into a readable string."""
    s = ""

    if mode == "short":
        s += f"DataFrame shape: {info_dict['shape']}\n"
        s += "First row\n" + "-" * 60 + "\n"
        s += info_dict["first_row"].to_string()

    elif mode == "sample":
        n_samples = len(info_dict["sample_rows"])
        s += f"DataFrame shape: {info_dict['shape']}\n"
        s += f"Columns: {', '.join(info_dict['columns'])}\n"
        s += f"\nRandom sample ({n_samples} rows)\n" + "-" * 60 + "\n"
        s += info_dict["sample_rows"].to_string()

    elif mode == "stats":
        s += f"DataFrame shape: {info_dict['shape']}\n"
        s += "\nStatistics\n" + "-" * 60 + "\n"

        numeric_stats = info_dict["numeric_stats"]
        if not numeric_stats.empty:
            s += "Numeric columns:\n"
            s += numeric_stats.to_string() + "\n"

        categorical_stats = info_dict["categorical_stats"]
        if categorical_stats:
            s += "\nCategorical columns:\n"
            for col, stats in categorical_stats.items():
                s += f"\n{col}:\n"
                s += stats["value_counts"].to_string()
                total_unique = stats["total_unique"]
                if total_unique > 5:
                    s += f"\n  ... ({total_unique - 5} more unique values)"
                s += "\n"

    return s
