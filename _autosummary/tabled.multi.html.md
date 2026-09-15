# tabled.multi

Multi-tabled data structures.

### Functions

| [`columns_of_all_tables`](#tabled.multi.columns_of_all_tables)(tables)                   | Return all columns from all tables, in order of first appearance.                            |
|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| [`columns_of_first_table`](#tabled.multi.columns_of_first_table)(tables)                  | Return the column names of the first table in `tables`.                                      |
| [`dataframes`](#tabled.multi.dataframes)(tables)                              | Cast to an iterable of dataframes.                                                           |
| [`execute_commands`](#tabled.multi.execute_commands)(commands, scope, ...[, ...])   | Carries `commands` operations out with tables taken from `scope`.                            |
| [`execute_table_commands`](#tabled.multi.execute_table_commands)(commands, tables[, ...]) | Run `commands` (`Load`/`Join`/`Remove`/`Rename`) against `tables`; see `execute_commands`.   |
| [`join_func`](#tabled.multi.join_func)(scope, command)                       | Interpreter for `Join`: inner-merge the table at `command.table_key` into `scope["cumul"]`.  |
| [`load_func`](#tabled.multi.load_func)(scope, command)                       | Interpreter for `Load`: set `scope["cumul"]` to `scope[command.key]`.                        |
| [`mapping_of_dataframes`](#tabled.multi.mapping_of_dataframes)(tables)                   | Cast to a mapping of dataframes                                                              |
| [`remove_func`](#tabled.multi.remove_func)(scope, command)                     | Interpreter for `Remove`: drop `command.fields` from `scope["cumul"]`.                       |
| [`rename_func`](#tabled.multi.rename_func)(scope, command)                     | Interpreter for `Rename`: rename `scope["cumul"]` columns and record the mapping in `scope`. |
| [`set_scope_value`](#tabled.multi.set_scope_value)(scope, key, value)              | Set `scope[key] = value` (in place).                                                         |

### Classes

| [`ColumnOrientedMapping`](#tabled.multi.ColumnOrientedMapping)(tables[, columns])   | A `{column_name: concatenated_column_values}` view over several tables.          |
|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`Join`](#tabled.multi.Join)(table_key)                            | Command: inner-join the accumulator with the table at `scope[table_key]`.        |
| [`Load`](#tabled.multi.Load)(key)                                  | Command: set the accumulator (`scope["cumul"]`) to `scope[key]`.                 |
| [`Remove`](#tabled.multi.Remove)(fields)                             | Command: drop `fields` (column or columns) from the accumulator.                 |
| [`Rename`](#tabled.multi.Rename)(rename_mapping)                     | Command: rename accumulator columns per `rename_mapping` (old name -> new name). |

### *class* tabled.multi.ColumnOrientedMapping(tables, columns=<function columns_of_first_table>)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

A `{column_name: concatenated_column_values}` view over several tables.

Keys are column names (by default, the columns of the first table);
each value is that column concatenated across all `tables`.

#### array(columns=None)

Concatenate a single column from all tables into one array.

`columns` must be a single column name (not a list): `.df(columns)`
then has to return a Series (not a DataFrame) for `.array` to work.

#### *property* columns

The columns that will be used in this mapping (the keys of the mapping)

#### columns_of_all_tables()

Return all columns from all tables, in order of first appearance.
This is useful for making a ColumnOrientedMapping without reverting to
the default columns argument, which is to use the columns of the first table.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Column`)]

#### columns_of_first_table()

Return the column names of the first table in `tables`.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Column`)]

#### df(columns=None)

Concatenate the given columns (all columns by default) from all tables into one dataframe.

### *class* tabled.multi.Join(table_key)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Command: inner-join the accumulator with the table at `scope[table_key]`.

### *class* tabled.multi.Load(key)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Command: set the accumulator (`scope["cumul"]`) to `scope[key]`.

### *class* tabled.multi.Remove(fields)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Command: drop `fields` (column or columns) from the accumulator.

### *class* tabled.multi.Rename(rename_mapping)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Command: rename accumulator columns per `rename_mapping` (old name -> new name).

### tabled.multi.columns_of_all_tables(tables)

Return all columns from all tables, in order of first appearance.
This is useful for making a ColumnOrientedMapping without reverting to
the default columns argument, which is to use the columns of the first table.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Column`)]

### tabled.multi.columns_of_first_table(tables)

Return the column names of the first table in `tables`.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`Column`)]

### tabled.multi.dataframes(tables)

Cast to an iterable of dataframes.

* **Return type:**
  [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[`DataFrame`]

### tabled.multi.execute_commands(commands, scope, interpreter_map, , extra_scope=None)

Carries `commands` operations out with tables taken from `scope`.

* **Parameters:**
  **commands** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)) – An iterable of join operations to carry out.

### tabled.multi.execute_table_commands(commands, tables, interpreter_map={<class 'tabled.multi.Join'>: <function join_func>, <class 'tabled.multi.Load'>: <function load_func>, <class 'tabled.multi.Remove'>: <function remove_func>, <class 'tabled.multi.Rename'>: <function rename_func>}, \*, extra_scope=None)

Run `commands` (`Load`/`Join`/`Remove`/`Rename`) against `tables`; see `execute_commands`.

### tabled.multi.join_func(scope, command)

Interpreter for `Join`: inner-merge the table at `command.table_key` into `scope["cumul"]`.

If `scope["renamed_columns"]` was set by a prior `Rename`, that column
renaming is applied to the joined table first, so a later rename stays
consistent across joins.

### tabled.multi.load_func(scope, command)

Interpreter for `Load`: set `scope["cumul"]` to `scope[command.key]`.

### tabled.multi.mapping_of_dataframes(tables)

Cast to a mapping of dataframes

* **Return type:**
  [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), `DataFrame`]

### tabled.multi.remove_func(scope, command)

Interpreter for `Remove`: drop `command.fields` from `scope["cumul"]`.

### tabled.multi.rename_func(scope, command)

Interpreter for `Rename`: rename `scope["cumul"]` columns and record the mapping in `scope`.

### tabled.multi.set_scope_value(scope, key, value)

Set `scope[key] = value` (in place).
