import pandas as pd
from tabled.base import DataframeKvReader


# Test cases
def test_DataframeKvReader():

    df = pd.DataFrame(
        {'A': [1, 2, 1], 'B': [4, 5, 4], 'C': [7, 8, 9], 'D': [10, 11, 12]}
    )

    kv_readers = [
        DataframeKvReader(df, ['A', 'B'], ['C', 'D']),
        DataframeKvReader(df.set_index(['A', 'B']), ['A', 'B'], ['C', 'D']),
        DataframeKvReader(df.set_index('A'), ['A', 'B'], ['C', 'D']),
    ]

    for kv_reader in kv_readers:
        # Accessing values
        key = (1, 4)
        expected_df = pd.DataFrame([{'C': 7, 'D': 10}, {'C': 9, 'D': 12}], index=[0, 2])
        assert (
            kv_reader[key]
            .reset_index(drop=True)
            .equals(expected_df.reset_index(drop=True))
        )
        assert list(kv_reader) == [(1, 4), (2, 5)]
