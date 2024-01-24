"""Multi-tabled data structures."""

from functools import cached_property
import pandas as pd
from typing import TypeVar, Callable, KT, Union, Iterable, Mapping, Dict

Column = TypeVar('Column')
TableKey = TypeVar('TableKey')
MappingOfDataFrames = Mapping[KT, pd.DataFrame]
DataFrames = Union[MappingOfDataFrames, Iterable[pd.DataFrame]]

# --------------------------------------------------------------------------------------
# Utils


def mapping_of_dataframes(tables: DataFrames) -> MappingOfDataFrames:
    """Cast to a mapping of dataframes"""
    if not isinstance(tables, Mapping):
        if isinstance(tables, Iterable):
            tables = dict(enumerate(tables))
        else:
            raise TypeError(f'Expected Mapping or Iterable, got {type(tables)}')
    return tables


def dataframes(tables: DataFrames) -> Iterable[pd.DataFrame]:
    """Cast to an iterable of dataframes."""
    if isinstance(tables, Mapping):
        tables = tables.values()
    return tables


# --------------------------------------------------------------------------------------
# Combined datasets
# See https://github.com/i2mint/tabled/discussions/3
from dataclasses import dataclass
from collections import ChainMap


def execute_commands(
    commands: Iterable, scope: Mapping, interpreter_map: Mapping, *, extra_scope=None,
):
    """
    Carries `commands` operations out with tables taken from `scope`.

    :param commands: An iterable of join operations to carry out.
    """
    if extra_scope is None:
        extra_scope = dict()
    _scope = ChainMap(extra_scope, scope)

    for command in commands:
        command_type = type(command)
        command_executor = interpreter_map.get(command_type, None)
        if command_executor is None:
            raise TypeError(f'Unknown command type: {type(command)}')
        yield command_executor(_scope, command)


# Define the JoinWith dataclass
# @dataclass
# class JoinWith:
#     table_key: str
#     remove: list = None


@dataclass
class Load:
    key: str


@dataclass
class Join:
    table_key: str


@dataclass
class Remove:
    fields: Union[str, Iterable[str]]


@dataclass
class Rename:
    rename_mapping: Dict[str, str]


def set_scope_value(scope, key, value):
    scope[key] = value


def load_func(scope, command):
    return set_scope_value(scope, 'cumul', scope[command.key])


def join_func(scope, command):
    table = scope[command.table_key].copy()
    if 'renamed_columns' in scope:
        for old_col, new_col in scope['renamed_columns'].items():
            if old_col in table:
                table.rename(columns={old_col: new_col}, inplace=True)
    cumul = scope['cumul']
    scope['cumul'] = cumul.merge(table, how='inner')


def remove_func(scope, command):
    scope['cumul'] = scope['cumul'].drop(columns=command.fields)


def rename_func(scope, command):
    scope['renamed_columns'] = command.rename_mapping
    for old_col, new_col in command.rename_mapping.items():
        scope['cumul'] = scope['cumul'].rename(columns={old_col: new_col})


dflt_tables_interpreter_map = {
    Load: load_func,
    Join: join_func,
    Remove: remove_func,
    Rename: rename_func,
}


def execute_table_commands(
    commands: Iterable,
    tables: Mapping[str, pd.DataFrame],
    interpreter_map: Mapping = dflt_tables_interpreter_map,
    *,
    extra_scope=None,
):
    return execute_commands(commands, tables, interpreter_map, extra_scope=extra_scope)


# def execute_commands(
#     commands: Iterable, tables: Mapping[str, pd.DataFrame]
# ) -> pd.DataFrame:
#     """
#     Carries `commands` operations out with tables taken from `tables`.

#     :param commands: An iterable of join operations to carry out.
#         Each join operation is either a table name (str) or a JoinWith object.
#         If it's a JoinWith object, it's assumed that the table has already been joined
#         and the fields to remove are in the `remove` attribute of the object.
#     :param tables: A mapping of table names to tables (pd.DataFrame)
#     """
#     # join_ops = map(ensure_join_op, resolution_sequence)
#     commands = iter(commands)
#     first_command = next(commands)
#     assert isinstance(first_command, Join)
#     table_key = first_command.table_key
#     cumul = tables[table_key]  # initialize my accumulator
#     for command in commands:
#         if isinstance(command, Join):
#             table = tables[command.table_key]
#             cumul = cumul.merge(table, how='inner')
#         elif isinstance(command, Remove):
#             cumul = cumul.drop(columns=command.fields)
#         else:
#             raise TypeError(f'Unknown command type: {type(command)}')
#     return cumul


# --------------------------------------------------------------------------------------
# Row-oriented mapping of multiple tables


# --------------------------------------------------------------------------------------
# Column-oriented mapping of multiple tables


def columns_of_first_table(tables: MappingOfDataFrames) -> Iterable[Column]:
    tables = mapping_of_dataframes(tables)
    first_table = next(iter(dataframes(tables)))
    return first_table.columns.values.tolist()


def columns_of_all_tables(tables: MappingOfDataFrames) -> Iterable[Column]:
    """Return all columns from all tables, in order of first appearance.
    This is useful for making a ColumnOrientedMapping without reverting to
    the default columns argument, which is to use the columns of the first table.
    """
    from itertools import chain

    tables = mapping_of_dataframes(tables)
    all_columns = chain.from_iterable(table.columns for table in tables.values())
    # return unique columns in order of first appearance
    return list(dict.fromkeys(all_columns))


class ColumnOrientedMapping(Mapping):
    def __init__(
        self,
        tables: MappingOfDataFrames,
        columns: Union[Callable, Iterable[Column]] = columns_of_first_table,
    ):
        self.tables = mapping_of_dataframes(tables)
        self._init_columns = columns

    @cached_property
    def columns(self):
        """The columns that will be used in this mapping (the keys of the mapping)"""
        if isinstance(self._init_columns, str):
            return [self._init_columns]
        elif callable(self._init_columns):
            return self._init_columns(self.tables)
        assert isinstance(
            self._init_columns, Iterable
        ), f'Expected Callable or Iterable, got {self._init_columns}'
        return self._init_columns

    @cached_property
    def _table_keys(self) -> Iterable[TableKey]:
        return list(self.tables.keys())

    def __iter__(self) -> Iterable[Column]:
        return iter(self.columns)

    def __getitem__(self, k):
        """Return a table with the given columns from all tables."""
        return self.df(k)

    def __len__(self):
        return len(self.columns)

    def __contains__(self, k):
        return k in self.columns

    def df(self, columns=None):
        """Concatinate all dataframes of given columns from all tables,
        returning a single dataframe."""
        if columns is None:
            columns = self.columns
        return pd.concat([table[columns] for table in dataframes(self.tables)])

    def array(self, columns=None):
        """Concatinate all the arrays of given columns from all tables,
        returning a single array."""
        return self.df(columns).array


# Convenience functions, placed in ColumnOrientedMapping for easy access
ColumnOrientedMapping.columns_of_first_table = columns_of_first_table
ColumnOrientedMapping.columns_of_all_tables = columns_of_all_tables
