"""
Utils
"""

from functools import partial
from typing import Mapping, KT, VT, Iterable, Callable, List
from collections import deque


def identity(x):
    return x


def split_keys(d):
    """Returns a dictionary where keys that had spaces were split into multiple keys

    Meant to be a convenience function for the user to use when they want to define a
    mapping where several keys map to the same value.

    >>> split_keys({'apple': 1, 'banana carrot': 2})
    {'apple': 1, 'banana': 2, 'carrot': 2}
    """
    return {split_k: v for k, v in d.items() for split_k in k.split()}


from typing import Dict, KT, Literal


def intersection_graph(
    sets: Dict[KT, set], edge_labels: Literal['elements', 'size', False] = False
):
    """
    A graph of all intersections between sets.
    (See https://en.wikipedia.org/wiki/Intersection_graph.)

    In graph theory, an adjacency list is a collection of sets used to represent a
    finite graph.
    Here, the vertices are the values of sets,
    and there is an edge between two vertices if the sets intersect.
    The weight of the edge is the size of the intersection.

    :param sets: A mapping of keys to sets of elements. These sets of elements will
        be the vertices of the graph.
    :param edge_labels: If 'elements', the edge labels are the elements of the intersection.
        If 'size', the edge labels are the size of the intersection.
        If False, there are no edge labels.
    :return: A graph, represented by an "adjacency list"
        (see https://en.wikipedia.org/wiki/Adjacency_list)
        (a dict whose keys are the keys of the input `sets` dict, and whose values
        tell us what sets of `sets` intersect with it),
        optionally with some information about this intersection.

    >>> sets = {
    ...     'A': {'b', 'c'},
    ...     'B': {'a', 'b', 'd', 'e', 'f'},
    ...     'C': {'f', 'g'},
    ...     'D': {'d', 'e', 'h', 'i'},
    ...     'E': {'i', 'j'}
    ... }
    >>> assert intersection_graph(sets) == {
    ...     'A': {'B'}, 'B': {'A', 'C', 'D'}, 'C': {'B'}, 'D': {'B', 'E'}, 'E': {'D'}
    ... }
    >>> assert intersection_graph(sets, edge_labels='elements') == {
    ...     'A': {'B': {'b'}},
    ...     'B': {'A': {'b'}, 'C': {'f'}, 'D': {'d', 'e'}},
    ...     'C': {'B': {'f'}},
    ...     'D': {'B': {'d', 'e'}, 'E': {'i'}},
    ...     'E': {'D': {'i'}}
    ... }
    >>> assert intersection_graph(sets, edge_labels='size') == {
    ...     'A': {'B': 1},
    ...     'B': {'A': 1, 'C': 1, 'D': 2},
    ...     'C': {'B': 1},
    ...     'D': {'B': 2, 'E': 1},
    ...     'E': {'D': 1}
    ... }

    """
    from collections import defaultdict
    from itertools import combinations

    graph = defaultdict(dict)
    for key1, key2 in combinations(sets.keys(), 2):
        intersection = sets[key1] & sets[key2]
        if intersection:
            graph[key1][key2] = intersection
            graph[key2][key1] = intersection
    if isinstance(edge_labels, str):
        if edge_labels == 'elements':
            return dict(graph)
        elif edge_labels == 'size':
            return map_values(map_values.len, graph)
    elif edge_labels is False:
        return map_values.set(graph)

    raise ValueError(f'Invalid value for edge_labels: {edge_labels}')


def map_values(func: Callable, d: dict):
    """Apply a function to all values of a dictionary.

    >>> map_values(lambda x: x ** 2, {1: 2, 3: 4})
    {1: 4, 3: 16}
    """
    return {k: func(v) for k, v in d.items()}


map_values.len = partial(map_values, len)
map_values.set = partial(map_values, set)
map_values.list = partial(map_values, list)


def invert_labeled_collection(
    d: Mapping[KT, Iterable[VT]], values_container: Callable = list
) -> Mapping[VT, Iterable[KT]]:
    """Invert a mapping whose values are iterables of objects,
    getting a mapping from objects to iterables of keys.

    >>> original_dict = {
    ...     "X": ['a', 'b'],
    ...     "Y": ['a'],
    ...     "Z": ['a', 'b', 'c']
    ... }
    >>> inverted_dict = invert_labeled_collection(original_dict)
    >>> inverted_dict
    {'a': ['X', 'Y', 'Z'], 'b': ['X', 'Z'], 'c': ['Z']}
    >>> invert_labeled_collection(inverted_dict)
    {'X': ['a', 'b'], 'Y': ['a'], 'Z': ['a', 'b', 'c']}

    The `values_container` argument can be used to cast the values of the inverted dict.

    >>> assert (
    ...     invert_labeled_collection(original_dict, values_container=set)
    ...     == {'a': {'X', 'Y', 'Z'}, 'b': {'X', 'Z'}, 'c': {'Z'}}
    ... )
    >>>
    >>> d = {'a': 'apple', 'b': 'banana'}
    >>> t = invert_labeled_collection(d, values_container=''.join)
    >>> t
    {'a': 'abbb', 'p': 'aa', 'l': 'a', 'e': 'a', 'b': 'b', 'n': 'bb'}
    >>> invert_labeled_collection(t, ''.join)
    {'a': 'apple', 'b': 'aaabnn'}

    """
    inverted_dict = {}
    for key, values in d.items():
        for value in values:
            inverted_dict.setdefault(value, []).append(key)
    if values_container is not list:
        inverted_dict = {k: values_container(v) for k, v in inverted_dict.items()}
    return inverted_dict


def breadth_first_traversal(graph: Mapping, start_node, *, yield_edges=False):
    """Yields nodes starting from the root node, expanding to neighbors recursively,
    using breadth-first search, without repeating nodes.

    :param graph: Adjacencies of the graph: A mapping from nodes to their neighbors.
    :param start_node: The node to start from (key of the graph adjacency mapping)
    :param yield_edges: If True, yield edges instead of nodes.
        The edges are yielded as tuples of (node, neighbor).

    >>> graph = {
    ...     'A': ['B'], 'B': ['A', 'C', 'D'], 'C': ['B'], 'D': ['B', 'E'], 'E': ['D']
    ... }
    >>> list(breadth_first_traversal(graph, 'B'))
    ['B', 'A', 'C', 'D', 'E']
    >>> list(breadth_first_traversal(graph, 'B', yield_edges=True))
    [('B', 'A'), ('B', 'C'), ('B', 'D'), ('D', 'E')]

    """

    visited = set()
    queue = deque([start_node])

    while queue:
        node = queue.popleft()
        if node not in visited:
            if yield_edges:
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        yield (node, neighbor)
            else:
                yield node
            visited.add(node)
            queue.extend(
                neighbor for neighbor in graph[node] if neighbor not in visited
            )


# -------------------------------------------------------------------------------------
# Expand and collapse rows and columns

import pandas as pd
from typing import List, Callable, Iterable, Union
import pandas as pd


def collapse_rows(
    df: pd.DataFrame, by: List[KT], *, container: Callable[[Iterable], Iterable] = list,
) -> pd.DataFrame:
    """
    Do a groupby to collapse (the rows of) a dataframe, gathering the other
    column's values (the ones that are not keys of the groupby) into lists.

    :param df: the dataframe to collapse
    :param by: the columns to group by (the keys of the groupby)
    :param container: the container to use to gather the other columns values

    >>> df = pd.DataFrame({
    ...     'a': [1, 1, 2, 2],
    ...     'b': [3, 4, 5, 6],
    ...     'c': [7, 8, 9, 10]
    ... })
    >>> df  # doctest: +NORMALIZE_WHITESPACE
       a  b   c
    0  1  3   7
    1  1  4   8
    2  2  5   9
    3  2  6  10
    >>> collapse_rows(df, ['a'])  # doctest: +NORMALIZE_WHITESPACE
       a       b        c
    0  1  [3, 4]   [7, 8]
    1  2  [5, 6]  [9, 10]

    """
    return df.groupby(by, as_index=False).agg(container)


def expand_rows(
    df: pd.DataFrame, grouped_columns: Union[str, List[KT]]
) -> pd.DataFrame:
    """
    Expands a DataFrame where specific columns were collapsed into containers back to its original form.
    Each column in `grouped_columns` should contain lists of the same length within each row.

    :param df: The DataFrame to expand.
    :param grouped_columns: The list of columns to expand
    :return: The expanded DataFrame.

    >>> df_collapsed = pd.DataFrame({
    ...     'a': [1, 2],
    ...     'b': [[3, 4], [5, 6, 66]],
    ...     'c': [[7, 8], [9, 10, 11]]
    ... })
    >>> expand_rows(df_collapsed, ['b', 'c'])  # doctest: +NORMALIZE_WHITESPACE
        a  b   c
    0  1  3   7
    1  1  4   8
    2  2  5   9
    3  2  6  10
    4  2  66  11
    """
    if isinstance(grouped_columns, str):
        grouped_columns = [grouped_columns]

    # Create a new list to hold the expanded rows
    expanded_rows = []

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Extract the lengths of lists in the grouped columns, assume all are the same length
        list_lengths = [len(row[col]) for col in grouped_columns]

        # Check if all list lengths are the same for safety
        if len(set(list_lengths)) != 1:
            raise ValueError(
                'All lists in a single row must have the same length to expand correctly.'
            )

        # Create individual rows for each index of the lists in grouped_columns
        for i in range(list_lengths[0]):
            # Start with a dict of non-grouped data
            new_row = {
                col: row[col] for col in df.columns if col not in grouped_columns
            }
            # Add the i-th element of each grouped column's list to the new row
            new_row.update({col: row[col][i] for col in grouped_columns})
            expanded_rows.append(new_row)

    # Create a new DataFrame from the expanded rows
    return pd.DataFrame(expanded_rows)


def collapse_columns(
    df: pd.DataFrame, groupings: Union[Dict[str, List[str]], Union[str, List[KT]]],
) -> pd.DataFrame:
    """
    Transforms specified columns of a dataframe into single columns where each row
    contains a dictionary of column names and values from the original dataframe.

    :param df: The dataframe to transform.
    :param groupings: A mapping that indicates which columns to collapse into dictionaries
        and what to call the new resulting column.
        If only a list of column names to be collapsed is given, it will be interpreted as
        the group_column_names in a single `{"collapsed": group_column_names}` dictionary,
        that is, all `group_column_names` are to be collapsed in to a single collapsed column.

    :return: A dataframe with the original columns not specified in `columns` untouched,
        and a new column `new_column_name` containing dictionaries of the collapsed columns.

    Example:

    >>> df = pd.DataFrame({
    ...     'a': [1, 1, 2, 2],
    ...     'b': [3, 4, 5, 6],
    ...     'c': [7, 8, 9, 10]
    ... })
    >>> df  # doctest: +NORMALIZE_WHITESPACE
       a  b   c
    0  1  3   7
    1  1  4   8
    2  2  5   9
    3  2  6  10
    >>> collapse_columns(df, {'ab': ['a', 'b']})  # doctest: +NORMALIZE_WHITESPACE
       c   ab
    0  7  {'a': 1, 'b': 3}
    1  8  {'a': 1, 'b': 4}
    2  9  {'a': 2, 'b': 5}
    3 10  {'a': 2, 'b': 6}
    """
    if isinstance(groupings, str):
        groupings = [groupings]
    if isinstance(groupings, list):
        # Convert list to a dictionary with default key if groupings is a list
        groupings = {'collapsed': groupings}

    # Copy the dataframe to avoid changing the original one
    result_df = df.copy()

    # Apply the grouping transformation
    for new_column, columns_to_collapse in groupings.items():
        # Ensure only valid columns are processed
        valid_columns = [col for col in columns_to_collapse if col in df.columns]

        if not valid_columns:
            raise ValueError(
                f'No valid columns found in {columns_to_collapse} to collapse into {new_column}.'
            )

        # Create a new column with dictionaries mapping old column names to their values
        result_df[new_column] = df[valid_columns].apply(
            lambda row: row.to_dict(), axis=1
        )

        # Remove the original columns that have been collapsed
        result_df.drop(columns=valid_columns, inplace=True)

    return result_df


def _column_dot_key_mapper(key, column_name):
    return f'{column_name}.{key}'


def _apply_key_mapper_to_keys(d: Union[dict, list], column_name, key_mapper):
    if isinstance(d, dict):
        return {key_mapper(key, column_name): value for key, value in d.items()}
    else:
        return {key_mapper(i, column_name): value for i, value in enumerate(d)}


def expand_columns(
    df: pd.DataFrame,
    expand_columns: Union[str, List[str]],
    *,
    drop=True,
    key_mapper=_column_dot_key_mapper,
) -> pd.DataFrame:
    """
    Expands the iterable values of specified columns in to new columns.
    The new columns will be named using the column_name and the key of the values of
    the iterable that is expanded (key if dict, integer index if sequence).

    :param df: The dataframe to transform.
    :param expand_columns: A list of column names whose values are dictionaries
        that need to be expanded into new columns.
    :param drop: Whether to drop the original columns that were expanded.
    :param key_mapper: A function that takes a key and a column name and returns a
        new key. By default, the new key is the concatenation of the column name and
        the original key. If None, will just take the original key. The reason for
        also taking the column_name by default is to avoid collisions if the keys are
        used in more than one column.

    :return: A dataframe with the expanded columns added.

    Examples:
    >>> df = pd.DataFrame({
    ...     'c': [7, 8, 9, 10],
    ...     'X': [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 5}, {'a': 2, 'b': 6}]
    ... })
    >>> expand_columns(df, ['X'])  # doctest: +NORMALIZE_WHITESPACE
       c  X.a  X.b
    0  7  1  3
    1  8  1  4
    2  9  2  5
    3 10  2  6

    Let's see what happens when the elements of an expanded column are lists instead of
    dicts, we ask to not drop, and we use `key_mapper=None`:

    >>> df = pd.DataFrame({
    ...     'c': [7, 8, 9, 10],
    ...     'X': [[1, 3], [1, 4], [2, 5], [2, 6]]
    ... })
    >>> expand_columns(df, ['X'], drop=False, key_mapper=None)  # doctest: +NORMALIZE_WHITESPACE
        c       X  0  1
    0   7  [1, 3]  1  3
    1   8  [1, 4]  1  4
    2   9  [2, 5]  2  5
    3  10  [2, 6]  2  6

    """
    if isinstance(expand_columns, str):
        expand_columns = [expand_columns]
    # Validate that all expand_columns are in df.columns
    for col in expand_columns:
        if col not in df.columns:
            raise ValueError(f'Column {col} does not exist in the DataFrame.')
    if key_mapper is None:
        key_mapper = lambda key, column_name: key

    # Copy the dataframe to avoid changing the original one
    result_df = df.copy()

    for col in expand_columns:
        _map_keys_of_dict = partial(
            _apply_key_mapper_to_keys, column_name=col, key_mapper=key_mapper
        )
        apply_func = lambda x: pd.Series(_map_keys_of_dict(x))
        # Extract dictionary to separate columns
        new_cols = result_df[col].apply(apply_func)
        # Merge new columns into the result DataFrame
        result_df = pd.concat([result_df, new_cols], axis=1)
        # Optionally drop the original column that was expanded
        if drop:
            result_df.drop(columns=[col], inplace=True)

    return result_df
