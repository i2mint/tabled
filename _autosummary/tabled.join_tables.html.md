# tabled.join_tables

Join multiple tables (pandas DataFrames) down to a target subset of columns.

Given a mapping of tables and the set of columns you want in the result, this
module figures out which pairs of tables to join, in what order, and which
overlapping fields to drop at each step, so the final result has exactly the
target columns.

Main entry points:

- `Join`: a join operation paired with optional fields to remove.
- `minimum_covering_tree`: the minimal tree of table joins covering the target subset.
- `generate_join_sequence`: the ordered sequence of `Join` operations to run.
- `compute_join_resolution`: carries out a join sequence and returns the result.

### Example

```pycon
>>> tables = {
...     'A': pd.DataFrame({'b': [1, 2, 3, 33], 'c': [4, 5, 6, 66]}),
...     'B': pd.DataFrame(
...         {
...             'b': [1, 2, 3],
...             'a': [4, 5, 6],
...             'd': [7, 8, 9],
...             'e': [10, 11, 12],
...             'f': [13, 14, 15],
...         }
...     ),
...     'C': pd.DataFrame({'f': [13, 14, 15], 'g': [4, 5, 6]}),
...     'D': pd.DataFrame(
...         {'d': [7, 8, 77], 'e': [10, 11, 77], 'h': [7, 8, 9], 'i': [1, 2, 3]}
...     ),
...     'E': pd.DataFrame({'i': [1, 2, 3], 'j': [4, 5, 6]}),
... }
>>> target_sub_set = {'b', 'g', 'j'}
>>> leaf_edges = get_leaf_edges(tables, target_sub_set)
>>> leaf_edges
[('B', 'C'), ('D', 'E')]
>>> join_sequence = generate_join_sequence(tables, leaf_edges, target_sub_set)
>>> join_sequence
['B', Join('C', remove=['a', 'f']), Join('D', remove=['d', 'e', 'h']), Join('E', remove=['i'])]
>>> join_result = compute_join_resolution(join_sequence, tables)
>>> join_result
   b  g  j
0  1  4  4
1  2  5  5
```

### Functions

| [`compute_join_resolution`](#tabled.join_tables.compute_join_resolution)(resolution_sequence, ...)   | Carries `resolution_sequence` join operations out with tables taken from `tables`.                                                                         |
|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`determine_remove_fields`](#tabled.join_tables.determine_remove_fields)(labeled_sets, ...)          | Determine which fields should be removed for a given table                                                                                                 |
| [`ensure_join_op`](#tabled.join_tables.ensure_join_op)(obj)                                 | Return `obj` if it is already a `Join`, else wrap it as `Join(obj)` (no fields removed).                                                                   |
| [`generate_join_sequence`](#tabled.join_tables.generate_join_sequence)(tables, leaf_edges, ...)     | Generate a sequence of joins with remove commands based on leaf edges                                                                                      |
| [`get_leaf_edges`](#tabled.join_tables.get_leaf_edges)(tables, target_subset[, ...])        | Return the covering-tree edges (see `minimum_covering_tree`) whose second table is a leaf (visited once).                                                  |
| [`minimum_covering_tree`](#tabled.join_tables.minimum_covering_tree)(tables, target_subset)        | Return the edges (pairs of table names) of a tree of joins covering `target_subset`.                                                                       |
| [`update_leaf_edges_after_removal`](#tabled.join_tables.update_leaf_edges_after_removal)(tables, ...)        | Update the list of leaf edges after removing an edge, ensuring that the resulting leaf edges do not lead to the loss of any elements in the target subset. |

### Classes

| [`Join`](#tabled.join_tables.Join)(table_id[, remove])   | A join step: which table to join in next, and which of its fields to drop after.   |
|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------|

### *class* tabled.join_tables.Join(table_id, remove=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A join step: which table to join in next, and which of its fields to drop after.

### tabled.join_tables.compute_join_resolution(resolution_sequence, tables)

Carries `resolution_sequence` join operations out with tables taken from `tables`.

* **Parameters:**
  * **resolution_sequence** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)) – An iterable of join operations to carry out.
    Each join operation is either a table name (str) or a Join object.
    If it’s a Join object, it’s assumed that the table has already been joined
    and the fields to remove are in the `remove` attribute of the object.
  * **tables** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), `DataFrame`]) – A mapping of table names to tables (pd.DataFrame)
* **Return type:**
  `DataFrame`

### tabled.join_tables.determine_remove_fields(labeled_sets, target_sub_set, joined_tables, current_table)

Determine which fields should be removed for a given table

* **Parameters:**
  * **labeled_sets** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`set`](https://docs.python.org/3/builtins/stdtypes.html#set)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – The sets of elements labeled by nodes
  * **target_sub_set** ([`set`](https://docs.python.org/3/builtins/stdtypes.html#set)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The target subset of elements that must remain covered
  * **joined_tables** ([`set`](https://docs.python.org/3/builtins/stdtypes.html#set)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The set of tables that have been or will be joined
  * **current_table** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The current table being processed
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]
* **Returns:**
  A list of fields to remove

### tabled.join_tables.ensure_join_op(obj)

Return `obj` if it is already a `Join`, else wrap it as `Join(obj)` (no fields removed).

### tabled.join_tables.generate_join_sequence(tables, leaf_edges, target_sub_set)

Generate a sequence of joins with remove commands based on leaf edges

* **Parameters:**
  * **leaf_edges** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – The list of leaf edges to process
  * **labeled_sets** – The sets of elements labeled by nodes
  * **target_sub_set** ([`set`](https://docs.python.org/3/builtins/stdtypes.html#set)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The target subset of elements that must remain covered
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`Join`](#tabled.join_tables.Join)]
* **Returns:**
  A list of Join operations

### tabled.join_tables.get_leaf_edges(tables, target_subset, start_node=None)

Return the covering-tree edges (see `minimum_covering_tree`) whose second table is a leaf (visited once).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`), [`TypeVar`](https://docs.python.org/3/library/typing.html#typing.TypeVar)(`KT`)]]

### tabled.join_tables.minimum_covering_tree(tables, target_subset, start_node=None)

Return the edges (pairs of table names) of a tree of joins covering `target_subset`.

Walks the tables’ column-overlap graph breadth-first from `start_node`
(or an arbitrary table if not given), accumulating edges until every
column in `target_subset` is covered by the tables seen so far.

### tabled.join_tables.update_leaf_edges_after_removal(tables, target_sub_set, current_leaf_edges)

Update the list of leaf edges after removing an edge, ensuring that the resulting
leaf edges do not lead to the loss of any elements in the target subset.

* **Parameters:**
  * **tables** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), `DataFrame`]) – A mapping of table names to tables (pd.DataFrame).
  * **target_sub_set** ([`set`](https://docs.python.org/3/builtins/stdtypes.html#set)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The target subset of columns that must remain covered.
  * **current_leaf_edges** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – The current list of leaf edges.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]
* **Returns:**
  An updated list of leaf edges.
