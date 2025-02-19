"""Test compare_tables.py."""

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from tabled.compare_tables import dataframe_diffs, InvalidComparison


def test_dataframe_diffs():
    # Test case 1: Different columns
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=[0, 1])
    df2 = pd.DataFrame({"A": [1, 2], "C": [5, 6]}, index=[0, 1])
    diffs = dataframe_diffs(df1, df2)
    assert "columns_diff" in diffs
    assert diffs["columns_diff"] == {"left_right": {"B"}, "right_left": {"C"}}

    # Test case 2: Different indices
    df1 = pd.DataFrame({"A": [1, 2]}, index=[0, 1])
    df2 = pd.DataFrame({"A": [1, 2]}, index=[1, 2])
    diffs = dataframe_diffs(df1, df2)
    assert "index_diff" in diffs
    assert diffs["index_diff"] == {"left_right": {0}, "right_left": {2}}

    # Test case 3: Different shapes
    df1 = pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2])
    df2 = pd.DataFrame({"A": [1, 2]}, index=[0, 1])
    diffs = dataframe_diffs(df1, df2)
    assert "shape_diff" in diffs
    assert diffs["shape_diff"] == ((3, 1), (2, 1))

    # Test case 4: Different data types
    df1 = pd.DataFrame({"A": [1, 2]}, index=[0, 1])
    df2 = pd.DataFrame({"A": [1.0, 2.0]}, index=[0, 1])
    diffs = dataframe_diffs(df1, df2)
    assert "dtypes_diff" in diffs
    assert diffs["dtypes_diff"] == {"A": (df1["A"].dtype, df2["A"].dtype)}

    # Test case 5: Different values
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=[0, 1])
    df2 = pd.DataFrame({"A": [1, 5], "B": [3, 4]}, index=[0, 1])
    diffs = dataframe_diffs(df1, df2)
    assert "columns_value_diff" in diffs
    assert "A" in diffs["columns_value_diff"]
    expected_diff = pd.DataFrame({"left": [2], "right": [5]}, index=[1], dtype=float)
    assert_frame_equal(
        diffs["columns_value_diff"]["A"], expected_diff, check_dtype=False
    )

    # Test case 6: No differences
    df1 = pd.DataFrame({"A": [1, 2]}, index=[0, 1])
    df2 = pd.DataFrame({"A": [1, 2]}, index=[0, 1])
    diffs = dataframe_diffs(df1, df2)
    assert diffs == {}

    # Test case 7: Non-unique indices
    df1 = pd.DataFrame({"A": [1, 2, 3]}, index=[0, 0, 1])
    df2 = pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 1])
    diffs = dataframe_diffs(df1, df2)
    assert isinstance(diffs["columns_value_diff"], InvalidComparison)

    # Test case 8: Custom comparison functions
    def custom_comparison(df1, df2):
        # Note we use != instead of == because we want to return True if the DataFrames are different
        # (By default, comparison values that resolve to False are not included in the output)
        return not df1.equals(df2)

    diffs = dataframe_diffs(df1, df2, comparisons=[custom_comparison])
    assert "custom_comparison" in diffs
    assert diffs["custom_comparison"]

    # Test case 9: Using diff_condition to filter results
    diffs = dataframe_diffs(df1, df2, diff_condition=lambda x: True)
    # All comparisons should be included regardless of their result
    assert "columns_diff" in diffs
    assert "index_diff" in diffs
    assert "shape_diff" in diffs
    assert "columns_value_diff" in diffs
    assert "dtypes_diff" in diffs

    # Test case 10: DataFrames with NaN values
    df1 = pd.DataFrame({"A": [1, 2, None]}, index=[0, 1, 2])
    df2 = pd.DataFrame({"A": [1, None, 3]}, index=[0, 1, 2])
    diffs = dataframe_diffs(df1, df2)
    assert "columns_value_diff" in diffs
    expected_diff = pd.DataFrame(
        {"left": [2.0, None], "right": [None, 3.0]}, index=[1, 2]
    )
    assert_frame_equal(
        diffs["columns_value_diff"]["A"], expected_diff, check_dtype=False
    )


test_dataframe_diffs()
