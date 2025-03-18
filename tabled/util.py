"""
Utils
"""

from functools import partial
from typing import Mapping, KT, VT, Iterable, Callable, List, Sized
from collections import deque


def identity(x):
    return x


def duplicate_groups(
    df, subset, *, output: str = "dataframe", keep_indices: bool = True
):
    """
    Get a DataFrame containing rows that have duplicate values for subset of columns.

    Args:
        df: Input DataFrame
        subset: Column name or list of column names to identify duplicates
        output: Output format, either "dataframe" or "series"
        keep_indices: If True (default), preserves the original index as a column named
                    by the index name or 'index' if unnamed. If False, keeps the
                    original index as the index of the result.

    Returns:
        Series with unique duplicate values as index and corresponding DataFrames as values
        or DataFrame with duplicated rows with the specified subset as index.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"A": [1, 1, 2, 3, 3], "B": ["a", "b", "c", "d", "e"]})
    >>> dups = duplicate_groups(df, "A")
    >>> dups  # doctest: +NORMALIZE_WHITESPACE
       B  index
    A
    1  a      0
    1  b      1
    3  d      3
    3  e      4
    >>> list(dups.index)
    [1, 1, 3, 3]
    >>> dups.loc[1].shape
    (2, 2)
    >>> dups = duplicate_groups(df, "A", output="series")
    >>> list(dups.index)
    [1, 3]
    >>> dups[1].shape
    (2, 2)
    >>> # Without keep_indices
    >>> dups_orig_idx = duplicate_groups(df, "A", keep_indices=False)
    >>> dups_orig_idx  # doctest: +NORMALIZE_WHITESPACE
       B
    A
    1  a
    1  b
    3  d
    3  e
    """
    # Find rows with duplicate values
    duplicated = df[df.duplicated(subset=subset, keep=False)]

    # Preserve the original index
    if output == "dataframe":
        # If subset is a string, use the column as index
        if isinstance(subset, str):
            result = duplicated.set_index(subset)

            # Preserve the original index as a column if requested
            if keep_indices:
                if df.index.name:
                    result[df.index.name] = duplicated.index
                else:
                    result["index"] = duplicated.index

            return result

        # For multi-column subset
        result = duplicated.set_index(subset)

        # Preserve the original index as a column if requested
        if keep_indices:
            if df.index.name:
                result[df.index.name] = duplicated.index
            else:
                result["index"] = duplicated.index

        return result

    elif output == "series":
        # Handle single column vs multiple columns
        if isinstance(subset, str):
            # For single column subset
            result = {}
            for value, group in duplicated.groupby(subset):
                # Preserve the original index in the group
                result[value] = group.copy()
            return pd.Series(result)
        else:
            # For multi-column subset
            result = {}
            for group_key, group in duplicated.groupby(subset):
                # If subset has multiple columns, use tuple as key
                key = group_key
                # Preserve the original index in the group
                result[key] = group.copy()
            return pd.Series(result)
    else:
        raise ValueError(
            f"Invalid output type: {output} (should be 'dataframe' or 'series')"
        )


def ensure_columns(df, columns=(), fill=None):
    """
    Ensure that a dataframe has certain columns, filling them with a certain value
    if they don't exist.
    """
    for c in columns:
        if c not in df.columns:
            df[c] = fill
    return df


def ensure_first_columns(df, columns=()):
    """
    Ensure that the given columns come first (if they exist), with the rest of the columns
    following in the order they were in the original dataframe.
    """
    first_columns = [c for c in columns if c in df.columns]
    new_column_order = first_columns + [c for c in df.columns if c not in first_columns]
    return df[new_column_order]


def ensure_last_columns(df, columns=()):
    """
    Ensure that the given columns come last (if they exist), with the rest of the columns
    preceding in the order they were in the original dataframe.
    """
    last_columns = [c for c in columns if c in df.columns]
    new_column_order = [c for c in df.columns if c not in last_columns] + last_columns
    return df[new_column_order]


def is_non_null_or_empty(value):
    """
    Check if a value is not None, not empty, and not an empty list.

    Often used with pandas dataframes to check if a cell is null or non-empty.

    ```
    num_of_non_empties_in_row = df.map(is_non_null_or_empty).sum(axis=1)
    num_of_non_empties_in_col = df.map(is_non_null_or_empty).sum(axis=0)
    ```

    And then you can do:

    ```
    num_of_non_empties_in_row.sort_values(ascending=False) to see which rows have the least empties (most actual data)
    ```

    """
    if isinstance(value, Sized):
        return len(value) > 0
    else:
        return pd.notnull(value)


def _isinstance(obj, class_or_tuple):
    """isinstance but not positional only arguments (so we can partial it)"""
    return isinstance(obj, class_or_tuple)


def is_instance_of(class_or_tuple):
    return partial(_isinstance, class_or_tuple=class_or_tuple)


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
    sets: Dict[KT, set], edge_labels: Literal["elements", "size", False] = False
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
        if edge_labels == "elements":
            return dict(graph)
        elif edge_labels == "size":
            return map_values(map_values.len, graph)
    elif edge_labels is False:
        return map_values.set(graph)

    raise ValueError(f"Invalid value for edge_labels: {edge_labels}")


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


# TODO: Integrate in tabled.base.dflt_ext_mapping
def auto_decode_bytes(
    b: bytes, *, try_first_bytes=(1e6, 1e7, 1e8), encoding: str = "utf-8", verbose=False
) -> str:
    """
    Decode a byte sequence into a string, trying charset_normalizer gueses if fails.

    This function attempts to decode the given bytes using the default encoding (usually 'utf-8').
    If that fails due to a `UnicodeDecodeError`, it uses `charset_normalizer` to detect the encoding
    by analyzing increasingly larger samples of the byte sequence, as specified in `try_first_bytes`.
    If all attempts fail, it analyzes the entire byte sequence to detect the encoding.

    Parameters:
        b (bytes): The byte sequence to decode.
        try_first_bytes (tuple of floats): Byte lengths to use for encoding detection samples.
            Defaults to (1e6, 1e7, 1e8).

    Returns:
        str: The decoded string.

    Raises:
        UnicodeDecodeError: If the byte sequence cannot be decoded after all attempts.

    Examples:
        >>> # Example with UTF-8 encoded bytes
        >>> s = 'Hello, world! Привет мир! こんにちは世界！'
        >>> b_utf8 = s.encode('utf-8')
        >>> auto_decode_bytes(b_utf8) == s
        True

        >>> # Example with UTF-16 encoded bytes
        >>> s_utf16 = 'Hello, world! 你好，世界！'
        >>> b_utf16 = s_utf16.encode('utf-16')
        >>> auto_decode_bytes(b_utf16) == s_utf16
        True

        Now, this is auto_decoding, but it doesn't mean it's robust.
        We use `charset_normalizer` to detect the encoding of the bytes, and then
        try to decode it with that encoding.
        But sometimes you can decode something that is not the original string,
        so be careful!!
        It's annoying to have to specify the encoding all the time, but this
        explicitness, and the errors that come with it, can be vital.

        Here are a few examples. We'll


        >>> s_latin1 = 'Héllo, wörld! Ça va?'
        >>> b_latin1 = s_latin1.encode('latin-1')  # latin-1 is ISO-8859-1
        >>> decoded_s = auto_decode_bytes(b_latin1, verbose=True)  # doctest: +ELLIPSIS
        Trying encoding: 'utf-8'
        Trying encoding: ...
        >>> decoded_s  # doctest: +SKIP
        'H幨lo, w顤ld! ド va?'
        >>> decoded_s == s_latin1  # doctest: +SKIP
        False

        (Note in the above that some tests were skipped. This is because the output
        is not deterministic and can vary depending on the system and the version of
        `charset_normalizer`.)

        >>> s_cp1252 = 'Special characters: € £ ¥ © ®'
        >>> b_cp1252 = s_cp1252.encode('cp1252')  # i.e. 'Windows-1252'
        >>> decoded_s = auto_decode_bytes(b_cp1252, verbose=True)
        Trying encoding: 'utf-8'
        Trying encoding: 'cp1125'
        >>> # See that charset_normalizer
        >>> decoded_s
        'Special characters: А г е й о'

    """

    import charset_normalizer  # pip install charset-normalizer

    if verbose:
        clog = lambda encoding: print(f"Trying encoding: '{encoding}'")
    else:
        clog = lambda encoding: None

    try:  # Try to decode using Python's default encoding (usually 'utf-8')
        clog(encoding)
        return b.decode(encoding)  # Use bytes.decode default encoding
    except UnicodeDecodeError:
        pass  # Proceed to detection steps

    import charset_normalizer  # pip install charset-normalizer

    try_first_bytes = list(map(int, try_first_bytes))

    for n_bytes in try_first_bytes:
        sample = b[:n_bytes]
        results = charset_normalizer.from_bytes(sample)
        if results:
            best_match = results.best()
            encoding = best_match.encoding
            try:
                clog(encoding)
                return b.decode(encoding=encoding)
            except UnicodeDecodeError:
                continue  # Try next sample size
        else:
            continue  # Detection failed, try next sample size

    # As a last resort, detect encoding using the entire byte sequence
    results = charset_normalizer.from_bytes(b)
    if results:
        encoding = results.best().encoding
        try:
            clog(encoding)
            return b.decode(encoding=encoding)
        except UnicodeDecodeError:
            pass  # Will raise error below

    raise UnicodeDecodeError("Unable to decode bytes with detected encodings.")


# -------------------------------------------------------------------------------------
# Expand and collapse rows and columns

import pandas as pd
from typing import List, Callable, Iterable, Union
import pandas as pd


def collapse_rows(
    df: pd.DataFrame,
    by: List[KT],
    *,
    container: Callable[[Iterable], Iterable] = list,
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
                "All lists in a single row must have the same length to expand correctly."
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
    df: pd.DataFrame,
    groupings: Union[Dict[str, List[str]], Union[str, List[KT]]],
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
        groupings = {"collapsed": groupings}

    # Copy the dataframe to avoid changing the original one
    result_df = df.copy()

    # Apply the grouping transformation
    for new_column, columns_to_collapse in groupings.items():
        # Ensure only valid columns are processed
        valid_columns = [col for col in columns_to_collapse if col in df.columns]

        if not valid_columns:
            raise ValueError(
                f"No valid columns found in {columns_to_collapse} to collapse into {new_column}."
            )

        # Create a new column with dictionaries mapping old column names to their values
        result_df[new_column] = df[valid_columns].apply(
            lambda row: row.to_dict(), axis=1
        )

        # Remove the original columns that have been collapsed
        result_df.drop(columns=valid_columns, inplace=True)

    return result_df


def column_sep_key_mapper(key, column_name, sep: str):
    return f"{column_name}{sep}{key}"


_column_dot_key_mapper = partial(column_sep_key_mapper, sep=".")


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
    drop_non_iterable_rows=False,
) -> pd.DataFrame:
    """
    Expands the iterable values of specified columns in to new columns.
    The new columns will be named using the column_name and the key of the values of
    the iterable that is expanded (key if dict, integer index if sequence).

    :param df: The dataframe to transform.
    :param expand_columns: A list of column names whose values are dictionaries
        that need to be expanded into new columns.
    :param drop: Whether to drop the original columns that were expanded.
    :param drop_non_iterable_rows: Whether to drop rows that have non-iterable values
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
            raise ValueError(f"Column {col} does not exist in the DataFrame.")
    if key_mapper is None:
        key_mapper = lambda key, column_name: key

    # Copy the dataframe to avoid changing the original one
    result_df = df.copy()

    # drop rows whose values (for exapnd_columns) are not iterable
    if drop_non_iterable_rows:
        for col in expand_columns:
            result_df = result_df[
                result_df[col].apply(lambda x: isinstance(x, Iterable))
            ]

    # Apply the expansion transformation
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


# -------------------------------------------------------------------------------------
# Serialization

import json
import datetime
import pandas as pd
import numpy as np


class PandasJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that can handle pandas and numpy types more robustly,
    even if they appear within nested data structures.

    >>> import json, datetime, pandas as pd, numpy as np
    >>> # Test with a DataFrame containing timestamps and missing values.
    >>> df = pd.DataFrame({
    ...     'a': [1, 2, 3],
    ...     'b': [pd.Timestamp('2023-04-09 00:02:53+0000', tz='UTC'),
    ...           pd.NaT,
    ...           pd.Timestamp('2023-04-09 00:02:53+0000', tz='UTC')]
    ... })
    >>> json_str = json.dumps(df, cls=PandasJSONEncoder)
    >>> json_str  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    '[{"a": 1, "b": "2023-04-09T00:02:53..."}..., {"a": 2, "b": null}, {"a": 3, "b": "2023-04-09T00:02:53..."}...]'

    >>> # Test with a Series containing timestamps and missing values.
    >>> s = pd.Series([pd.Timestamp('2023-04-09 00:02:53+0000', tz='UTC'), pd.NaT])
    >>> json_str = json.dumps(s, cls=PandasJSONEncoder)
    >>> json_str  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    '{"0": "2023-04-09T00:02:53...", "1": null}'

    >>> # Test with numpy arrays and numpy scalar types.
    >>> data = {
    ...     "arr": np.array([1, 2, 3], dtype=np.int32),
    ...     "flt": np.float32(3.14),
    ...     "bool": np.bool_(False)
    ... }
    >>> json_str = json.dumps(data, cls=PandasJSONEncoder)
    >>> json_str  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    '{"arr": [1, 2, 3], "flt": 3.14..., "bool": false}'

    >>> # Test with a datetime.date.
    >>> date_val = datetime.date(2002, 1, 1)
    >>> json.dumps(date_val, cls=PandasJSONEncoder)  # doctest: +NORMALIZE_WHITESPACE
    '"2002-01-01"'
    """

    def default(self, obj):
        # Handle pandas DataFrame by delegating to its own JSON conversion.
        if isinstance(obj, pd.DataFrame):
            # Using 'records' orientation to produce a list of row dictionaries.
            # Pandas will also handle nested types like Timestamps.
            return json.loads(obj.to_json(orient="records", date_format="iso"))
        # Handle pandas Series similarly.
        if isinstance(obj, pd.Series):
            # to_json for Series returns a JSON object (dict) keyed by the index.
            # This approach ensures that any non-JSON-serializable objects are handled by pandas.
            return json.loads(obj.to_json(date_format="iso"))
        # Handle pandas Timestamp objects.
        if isinstance(obj, pd.Timestamp):
            if pd.isna(obj):
                return None
            return obj.isoformat()
        # Convert numpy arrays to lists.
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Convert numpy boolean, floating, and integer scalars to native Python types.
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        # Convert datetime.date and datetime.datetime to ISO 8601 strings.
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        # For other objects, if pd.isna returns a boolean True, return None.
        is_na = pd.isna(obj)
        if isinstance(is_na, bool) and is_na:
            return None
        # Fallback to the default method.
        return super().default(obj)


pandas_json_dumps = partial(json.dumps, cls=PandasJSONEncoder)
