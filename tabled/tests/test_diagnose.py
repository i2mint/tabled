"""Tests for the diagnose module."""

import pytest
import pandas as pd
from tabled.diagnose import (
    dataframe_info,
    diagnose_table_collection,
    print_dataframe_info,
    register_info_func,
    list_info_funcs,
    DFLT_INFO_FUNCS,
    scalar_columns,
    _get_shape,
    _get_columns,
    _get_first_row,
    _get_sample_rows,
    _get_numeric_stats,
    _get_categorical_stats,
)


class TestScalarColumns:
    """Tests for the scalar_columns function."""

    def test_scalar_columns_basic(self):
        df = pd.DataFrame(
            {
                'A': [1, 2, 3],
                'B': ['x', 'y', 'z'],
                'C': [{'a': 1}, {'b': 2}, {'c': 3}],  # Non-serializable
                'D': [[1, 2], [3, 4], [5, 6]],  # Non-serializable
            }
        )
        result = scalar_columns(df)
        assert set(result) == {'A', 'B'}

    def test_scalar_columns_all_scalar(self):
        df = pd.DataFrame(
            {
                'int_col': [1, 2, 3],
                'float_col': [1.1, 2.2, 3.3],
                'str_col': ['a', 'b', 'c'],
                'bool_col': [True, False, True],
            }
        )
        result = scalar_columns(df)
        assert set(result) == {'int_col', 'float_col', 'str_col', 'bool_col'}

    def test_scalar_columns_empty_dataframe(self):
        df = pd.DataFrame()
        result = scalar_columns(df)
        assert result == []


class TestInfoFunctions:
    """Tests for individual info functions."""

    def setup_method(self):
        self.df = pd.DataFrame(
            {
                'a': [1, 2, 3, 4, 5],
                'b': ['x', 'y', 'z', 'x', 'y'],
                'c': [1.1, 2.2, 3.3, 4.4, 5.5],
            }
        )

    def test_get_shape(self):
        assert _get_shape(self.df) == (5, 3)

    def test_get_columns(self):
        assert _get_columns(self.df) == ['a', 'b', 'c']

    def test_get_first_row(self):
        first_row = _get_first_row(self.df)
        assert first_row['a'] == 1
        assert first_row['b'] == 'x'
        assert first_row['c'] == 1.1

    def test_get_first_row_empty_df(self):
        empty_df = pd.DataFrame()
        first_row = _get_first_row(empty_df)
        assert isinstance(first_row, pd.Series)
        assert len(first_row) == 0

    def test_get_sample_rows(self):
        sample = _get_sample_rows(self.df, n_samples=3)
        assert len(sample) == 3
        assert list(sample.columns) == ['a', 'b', 'c']

    def test_get_sample_rows_more_than_available(self):
        small_df = pd.DataFrame({'x': [1, 2]})
        sample = _get_sample_rows(small_df, n_samples=5)
        assert len(sample) == 2

    def test_get_sample_rows_empty_df(self):
        empty_df = pd.DataFrame()
        sample = _get_sample_rows(empty_df)
        assert len(sample) == 0

    def test_get_numeric_stats(self):
        stats = _get_numeric_stats(self.df)
        assert 'a' in stats.columns
        assert 'c' in stats.columns
        assert 'b' not in stats.columns  # string column excluded
        assert stats.loc['mean', 'a'] == 3.0

    def test_get_numeric_stats_no_numeric_columns(self):
        str_df = pd.DataFrame({'x': ['a', 'b', 'c']})
        stats = _get_numeric_stats(str_df)
        assert stats.empty

    def test_get_categorical_stats(self):
        stats = _get_categorical_stats(self.df)
        assert 'b' in stats
        assert 'a' not in stats  # numeric column excluded
        assert 'c' not in stats  # numeric column excluded

        b_stats = stats['b']
        assert 'value_counts' in b_stats
        assert 'total_unique' in b_stats
        assert b_stats['total_unique'] == 3


class TestDataframeInfo:
    """Tests for the dataframe_info function."""

    def setup_method(self):
        self.df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})

    def test_dataframe_info_default(self):
        info = dataframe_info(self.df)
        assert 'shape' in info
        assert 'columns' in info
        assert info['shape'] == (3, 2)
        assert info['columns'] == ['a', 'b']

    def test_dataframe_info_custom_funcs(self):
        custom_funcs = {'shape': _get_shape, 'num_cols': lambda df: len(df.columns)}
        info = dataframe_info(self.df, custom_funcs)
        assert set(info.keys()) == {'shape', 'num_cols'}
        assert info['shape'] == (3, 2)
        assert info['num_cols'] == 2

    def test_dataframe_info_custom_egress(self):
        custom_funcs = {'shape': _get_shape}
        info_list = dataframe_info(self.df, custom_funcs, egress=list)
        assert isinstance(info_list, list)
        assert len(info_list) == 1
        assert info_list[0] == ('shape', (3, 2))


class TestDiagnoseTableCollection:
    """Tests for the diagnose_table_collection function."""

    def setup_method(self):
        self.tables = {
            'table1': pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']}),
            'table2': pd.DataFrame({'c': [3, 4, 5], 'd': ['z', 'w', 'v']}),
        }

    def test_diagnose_table_collection_default(self):
        diagnosis = diagnose_table_collection(self.tables)
        assert set(diagnosis.keys()) == {'table1', 'table2'}

        table1_info = diagnosis['table1']
        assert 'shape' in table1_info
        assert table1_info['shape'] == (2, 2)

        table2_info = diagnosis['table2']
        assert table2_info['shape'] == (3, 2)

    def test_diagnose_table_collection_custom_egress(self):
        diagnosis_list = diagnose_table_collection(self.tables, egress=list)
        assert isinstance(diagnosis_list, list)
        assert len(diagnosis_list) == 2

        # Check that each item is a (table_key, info_dict) tuple
        for item in diagnosis_list:
            assert isinstance(item, tuple)
            assert len(item) == 2
            table_key, info_dict = item
            assert table_key in self.tables
            assert 'shape' in info_dict


class TestPrintDataframeInfo:
    """Tests for backward compatibility with print_dataframe_info."""

    def setup_method(self):
        self.df = pd.DataFrame(
            {
                'a': [1, 2, 3, 4, 5],
                'b': ['x', 'y', 'z', 'x', 'y'],
                'c': [1.1, 2.2, 3.3, 4.4, 5.5],
            }
        )

    def test_print_dataframe_info_short_mode(self):
        result = print_dataframe_info(self.df, mode='short', egress=None)
        assert 'shape: (5, 3)' in result
        assert 'First row' in result

    def test_print_dataframe_info_sample_mode(self):
        result = print_dataframe_info(self.df, mode='sample', egress=None)
        assert 'shape: (5, 3)' in result
        assert 'Columns: a, b, c' in result
        assert 'Random sample' in result

    def test_print_dataframe_info_stats_mode(self):
        result = print_dataframe_info(self.df, mode='stats', egress=None)
        assert 'shape: (5, 3)' in result
        assert 'Statistics' in result
        assert 'Numeric columns' in result
        assert 'Categorical columns' in result

    def test_print_dataframe_info_exclude_columns(self):
        result = print_dataframe_info(
            self.df, exclude_columns=['b'], mode='sample', egress=None
        )
        assert 'Columns: a, c' in result
        assert 'b' not in result

    def test_print_dataframe_info_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            print_dataframe_info(self.df, mode='invalid')


class TestRegistrationFunctions:
    """Tests for the registration functionality."""

    def setup_method(self):
        # Store original state
        self.original_funcs = DFLT_INFO_FUNCS.copy()

    def teardown_method(self):
        # Restore original state
        DFLT_INFO_FUNCS.clear()
        DFLT_INFO_FUNCS.update(self.original_funcs)

    def test_register_info_func(self):
        def custom_func(df):
            return len(df.columns) * 2

        original_count = len(DFLT_INFO_FUNCS)
        register_info_func('double_cols', custom_func)

        assert len(DFLT_INFO_FUNCS) == original_count + 1
        assert 'double_cols' in DFLT_INFO_FUNCS
        assert DFLT_INFO_FUNCS['double_cols'] == custom_func

    def test_register_info_func_overwrite_false(self):
        with pytest.raises(ValueError, match="already exists"):
            register_info_func('shape', lambda df: 'new')

    def test_register_info_func_overwrite_true(self):
        new_func = lambda df: 'overwritten'
        register_info_func('shape', new_func, overwrite=True)
        assert DFLT_INFO_FUNCS['shape'] == new_func

    def test_list_info_funcs(self):
        funcs = list_info_funcs()
        expected_funcs = {
            'shape',
            'columns',
            'first_row',
            'sample_rows',
            'numeric_stats',
            'categorical_stats',
        }
        assert set(funcs) == expected_funcs

    def test_registered_func_in_dataframe_info(self):
        def memory_usage(df):
            return df.memory_usage(deep=True).sum()

        register_info_func('memory', memory_usage)

        df = pd.DataFrame({'a': [1, 2, 3]})
        info = dataframe_info(df)

        assert 'memory' in info
        assert isinstance(
            info['memory'], (int, type(info['memory']))
        )  # Accept both Python int and numpy int types
        assert info['memory'] > 0


class TestIntegration:
    """Integration tests for the diagnose module."""

    def test_end_to_end_workflow(self):
        # Create sample data
        df1 = pd.DataFrame(
            {
                'id': [1, 2, 3],
                'name': ['Alice', 'Bob', 'Charlie'],
                'score': [85.5, 92.3, 78.1],
            }
        )

        df2 = pd.DataFrame({'product': ['A', 'B'], 'price': [10.99, 15.49]})

        tables = {'users': df1, 'products': df2}

        # Test collection diagnosis
        diagnosis = diagnose_table_collection(tables)

        assert 'users' in diagnosis
        assert 'products' in diagnosis
        assert diagnosis['users']['shape'] == (3, 3)
        assert diagnosis['products']['shape'] == (2, 2)

        # Test individual table analysis
        user_info = dataframe_info(
            df1,
            {
                'shape': _get_shape,
                'columns': _get_columns,
                'numeric_cols': lambda df: len(
                    df.select_dtypes(include='number').columns
                ),
            },
        )

        assert user_info['shape'] == (3, 3)
        assert user_info['columns'] == ['id', 'name', 'score']
        assert user_info['numeric_cols'] == 2  # id and score
