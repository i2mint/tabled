"""Tools to compare tables"""

from typing import Callable, Sequence, Union, Any
import pandas as pd

ComparisonName = str
ComparisonFunc = Callable[[pd.DataFrame, pd.DataFrame], Any]
ComparisonsDict = dict[ComparisonName, ComparisonFunc]
ComparisonsList = Sequence[Union[ComparisonName, ComparisonFunc]]
Comparisons = Union[ComparisonsDict, ComparisonsList, ComparisonName, ComparisonFunc]


class BinaryFuncResult(dict):
    def __bool__(self):
        return any(self.values())

    @classmethod
    def from_func(cls, func, x, y):
        return cls(
            left_right=func(x, y),
            right_left=func(y, x),
        )


def columns_diff(df1: pd.DataFrame, df2: pd.DataFrame):
    """Return columns that are not common between df1 and df2."""
    return BinaryFuncResult.from_func(
        lambda x, y: x - y, set(df1.columns), set(df2.columns)
    )


def index_diff(df1: pd.DataFrame, df2: pd.DataFrame):
    """Return indices that are not common between df1 and df2."""
    return BinaryFuncResult.from_func(
        lambda x, y: x - y, set(df1.index), set(df2.index)
    )


def shape_diff(df1: pd.DataFrame, df2: pd.DataFrame):
    """Return the shapes of df1 and df2 if they differ."""
    df1_shape = df1.shape
    df2_shape = df2.shape
    if df1_shape != df2_shape:
        return df1_shape, df2_shape


def columns_value_diff(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    For each column present in both dataframes, compare the values row-wise.
    Returns a dict with column names as keys and DataFrames of differences as values.
    Only columns with differences are included.
    """
    if not df1.index.is_unique or not df2.index.is_unique:
        return InvalidComparison("Indices are not unique.")
    common_index = df1.index.intersection(df2.index)
    common_columns = df1.columns.intersection(df2.columns)
    diffs = {}
    for col in common_columns:
        s1 = df1.loc[common_index, col]
        s2 = df2.loc[common_index, col]
        diff_s = s1.compare(s2, keep_shape=False, keep_equal=False)
        if not diff_s.empty:
            diff_s = diff_s.rename(columns={"self": "left", "other": "right"})
            diffs[col] = diff_s
    return diffs


def dtypes_diff(df1: pd.DataFrame, df2: pd.DataFrame):
    """Return columns where the data types differ between df1 and df2."""
    common_columns = df1.columns.intersection(df2.columns)
    diffs = {}
    for col in common_columns:
        dtype1 = df1[col].dtype
        dtype2 = df2[col].dtype
        if dtype1 != dtype2:
            diffs[col] = (dtype1, dtype2)
    return diffs


class InvalidComparison(ValueError):
    """Raised or returned when a comparison that was asked for is not applicable to the dataframes in question"""


DFLT_COMPARISONS: ComparisonsDict = {
    "columns_diff": columns_diff,
    "index_diff": index_diff,
    "shape_diff": shape_diff,
    "columns_value_diff": columns_value_diff,
    "dtypes_diff": dtypes_diff,
    # ... insert more here
}


def ensure_comparisons_dict(comparisons: Comparisons) -> ComparisonsDict:
    """Ensure that the comparisons are in the form of a dictionary."""
    if isinstance(comparisons, dict):
        return comparisons
    elif isinstance(comparisons, (str, Callable)):
        comparisons = [comparisons]

    def comparison_name_and_func(comparisons):
        for comparison in comparisons:
            if isinstance(comparison, str):
                yield comparison, DFLT_COMPARISONS[comparison]
            else:
                yield comparison.__name__, comparison

    return dict(comparison_name_and_func(comparisons))


def dataframe_diffs(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    comparisons: Comparisons = DFLT_COMPARISONS,
    *,
    diff_condition=bool
) -> dict:
    """
    Compare the diff of dataframes using specified diff comparison functions.

    Returns a dictionary with comparison names as keys and comparison results as values.

    Parameters:
        df1 (pd.DataFrame): The first dataframe to compare.
        df2 (pd.DataFrame): The second dataframe to compare.
        comparisons (Comparisons): A dictionary or list of comparison functions or names.
            Defaults to DFLT_COMPARISONS.
        diff_condition (callable): A function that determines whether to include a comparison result
            in the output dictionary based on the comparison result.
            Defaults to the built-in `bool` function.

    Returns:
        dict: A dictionary with comparison names as keys and comparison results as values.

    Example:
        >>> import pandas as pd
        >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=[0, 1])
        >>> df2 = pd.DataFrame({'A': [1, 2], 'C': [5, 6]}, index=[1, 2])
        >>> diffs = dataframe_diffs(df1, df2)
        >>> diffs  # doctest: +NORMALIZE_WHITESPACE
        {'columns_diff': {'left_right': {'B'}, 'right_left': {'C'}},
         'index_diff': {'left_right': {0}, 'right_left': {2}},
         'columns_value_diff': {'A':    left  right
        1     2      1}}
    """
    comparisons = ensure_comparisons_dict(comparisons)
    diffs = {}
    for name, comparison in comparisons.items():
        diff_value = comparison(df1, df2)
        if diff_condition(diff_value):
            diffs[name] = diff_value
    return diffs
