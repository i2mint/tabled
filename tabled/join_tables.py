"""
This module provides functionality for joining multiple tables (pandas DataFrames) based on a target subset of columns.
It includes classes and functions to determine the optimal sequence of joins and the fields to remove during the join process.
Classes:
    Join: Represents a join operation with optional fields to remove.
Functions:
    minimum_covering_tree(tables, target_subset, start_node=None):
        Computes the minimum covering tree for the given tables and target subset of columns.
    get_leaf_edges(tables, target_subset, start_node=None):
        Retrieves the leaf edges of the minimum covering tree for the given tables and target subset of columns.
    update_leaf_edges_after_removal(tables, target_sub_set, current_leaf_edges):
        Updates the list of leaf edges after removing an edge, ensuring that the resulting leaf edges do not lead to the loss of any elements in the target subset.
    determine_remove_fields(labeled_sets, target_sub_set, joined_tables, current_table):
        Determines which fields should be removed for a given table to ensure the target subset remains covered.
    generate_join_sequence(tables, leaf_edges, target_sub_set):
        Generates a sequence of joins with remove commands based on leaf edges and the target subset of columns.
    ensure_join_op(obj):
        Ensures that the given object is a Join instance.
    compute_join_resolution(resolution_sequence, tables):
        Carries out the join operations specified in the resolution sequence with the given tables.

Example:
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

"""

from typing import KT, VT, Literal, Iterable, Callable, List, Tuple, Set, Mapping, Dict
import pandas as pd
from collections import deque
from tabled.util import (
    intersection_graph,
    invert_labeled_collection,
    map_values,
    breadth_first_traversal,
)


class Join:
    def __init__(self, table_id: str, remove: List[str] = None):
        self.table_id = table_id
        self.remove = remove or []

    def __repr__(self):
        # note: sorting self.remove so doctests are consistent
        remove_str = f", remove={sorted(self.remove)}" if self.remove else ""
        return f"Join('{self.table_id}'{remove_str})"

    def __eq__(self, other):
        if not isinstance(other, Join):
            return False
        return self.table_id == other.table_id and set(self.remove) == set(other.remove)


def minimum_covering_tree(
    tables: Mapping[str, pd.DataFrame],
    target_subset: Iterable[VT],
    start_node=None,
):

    labeled_sets = {table_id: set(df.columns) for table_id, df in tables.items()}
    intersections = intersection_graph(labeled_sets, edge_labels="elements")
    graph = {k: list(v) for k, v in intersections.items()}
    if start_node is None:  # if not start_node is given
        start_node = next(iter(graph))  # take the first node
    target_sub_set = set(target_subset)
    traversal = breadth_first_traversal(graph, start_node, yield_edges=True)

    edges = list()
    covered = set()

    for set_label_1, set_label_2 in traversal:
        if target_sub_set.issubset(covered):
            break
        covered |= labeled_sets[set_label_1] | labeled_sets[set_label_2]
        edges.append((set_label_1, set_label_2))

    return edges


def get_leaf_edges(
    tables: Mapping[str, pd.DataFrame], target_subset: Set[VT], start_node: KT = None
) -> List[Tuple[KT, KT]]:
    labeled_sets = {table_id: set(df.columns) for table_id, df in tables.items()}
    intersections = intersection_graph(labeled_sets, edge_labels="elements")
    graph = {k: list(v) for k, v in intersections.items()}
    if start_node is None:  # if no start_node is given
        start_node = next(iter(graph))  # take the first node

    traversal = breadth_first_traversal(graph, start_node, yield_edges=True)

    edges = []  # List to store edges of the covering tree
    covered = set()  # Set to store covered elements
    node_connections = {}  # Dictionary to track the number of connections for each node

    for node1, node2 in traversal:
        if target_subset.issubset(covered):
            break

        # Update covered elements and the number of connections for each node
        covered |= labeled_sets[node1] | labeled_sets[node2]
        node_connections[node1] = node_connections.get(node1, 0) + 1
        node_connections[node2] = node_connections.get(node2, 0) + 1
        edges.append((node1, node2))

    # Filter edges to return only those that lead to a node with one connection
    leaf_edges = [edge for edge in edges if node_connections[edge[1]] == 1]

    return leaf_edges


def update_leaf_edges_after_removal(
    tables: Mapping[str, pd.DataFrame],
    target_sub_set: Set[str],
    current_leaf_edges: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """
    Update the list of leaf edges after removing an edge, ensuring that the resulting
    leaf edges do not lead to the loss of any elements in the target subset.

    :param graph: The graph represented as an adjacency list.
    :param labeled_sets: The sets of elements labeled by nodes.
    :param target_subset: The target subset of elements that must remain covered.
    :param current_leaf_edges: The current list of leaf edges.
    :return: An updated list of leaf edges.
    """
    new_leaf_edges = []
    labeled_sets = {table_id: set(df.columns) for table_id, df in tables.items()}
    intersections = intersection_graph(labeled_sets, edge_labels="elements")
    graph = {k: list(v) for k, v in intersections.items()}
    for node1, node2 in current_leaf_edges:
        # Use symmetric difference to consider the edge as undirected
        unique_elements = (
            (labeled_sets[node1] - labeled_sets[node2])
            | (labeled_sets[node2] - labeled_sets[node1])
        ) & target_sub_set

        # Check if the unique elements do not include any from the target subset
        if not unique_elements:
            # Remove the edge from the graph
            graph[node1].remove(node2)
            graph[node2].remove(node1)

        else:
            # If the edge cannot be removed, add it to the new list of leaf edges
            new_leaf_edges.append((node1, node2))

    # Recalculate the leaf edges after removals
    for node, neighbors in graph.items():
        if len(neighbors) == 1:  # If the node is now a leaf node
            neighbor = neighbors[0]
            if (node, neighbor) not in new_leaf_edges and (
                neighbor,
                node,
            ) not in new_leaf_edges:
                new_leaf_edges.append((node, neighbor))

    return new_leaf_edges


def determine_remove_fields(
    labeled_sets: Dict[str, Set[str]],
    target_sub_set: Set[str],
    joined_tables: Set[str],
    current_table: str,
) -> List[str]:
    """
    Determine which fields should be removed for a given table

    :param labeled_sets: The sets of elements labeled by nodes
    :param target_sub_set: The target subset of elements that must remain covered
    :param joined_tables: The set of tables that have been or will be joined
    :param current_table: The current table being processed
    :return: A list of fields to remove
    """
    fields = labeled_sets[current_table]
    removable_fields = [
        field
        for field in fields
        if field not in target_sub_set
        and all(
            field not in labeled_sets[table]
            for table in joined_tables
            if table != current_table
        )
    ]
    return removable_fields


def generate_join_sequence(
    tables: Mapping[str, pd.DataFrame],
    leaf_edges: List[Tuple[str, str]],
    target_sub_set: Set[str],
) -> List[Join]:
    """
    Generate a sequence of joins with remove commands based on leaf edges

    :param leaf_edges: The list of leaf edges to process
    :param labeled_sets: The sets of elements labeled by nodes
    :param target_sub_set: The target subset of elements that must remain covered
    :return: A list of Join operations
    """
    labeled_sets = {table_id: set(df.columns) for table_id, df in tables.items()}
    joins = []  # Stores the sequence of joins
    joined_tables = set()  # Tracks tables that have been joined
    future_joins = {
        node for edge in leaf_edges for node in edge
    }  # Tracks all future joins

    # Start with the first node of the first edge without removals
    first_node = leaf_edges[0][0]
    first_node_remove_fields = determine_remove_fields(
        labeled_sets, target_sub_set, future_joins, first_node
    )
    joins.append(first_node)
    for node1, node2 in leaf_edges:
        for node in (node1, node2):
            if node not in joined_tables:
                # Determine the fields to remove considering future joins
                remove_fields = determine_remove_fields(
                    labeled_sets, target_sub_set, future_joins, node
                )
                if node == first_node:
                    remove_fields = first_node_remove_fields
                elif (
                    node in leaf_edges[0]
                ):  # If the node is to join with the first node
                    remove_fields.extend(
                        first_node_remove_fields
                    )  # Append remove_fields from the first node

                if node != first_node:
                    joins.append(Join(node, remove_fields))
                # Update the tracking sets
                joined_tables.add(node)
                future_joins.remove(node)

    return joins


def ensure_join_op(obj):
    if not isinstance(obj, Join):
        return Join(obj)
    return obj


def compute_join_resolution(
    resolution_sequence: Iterable, tables: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Carries `resolution_sequence` join operations out with tables taken from `tables`.

    :param resolution_sequence: An iterable of join operations to carry out.
        Each join operation is either a table name (str) or a Join object.
        If it's a Join object, it's assumed that the table has already been joined
        and the fields to remove are in the `remove` attribute of the object.
    :param tables: A mapping of table names to tables (pd.DataFrame)
    """
    join_ops = map(ensure_join_op, resolution_sequence)
    table_key = next(join_ops).table_id
    joined = tables[table_key]
    for join_op in join_ops:
        table = tables[join_op.table_id]
        joined = joined.merge(table, how="inner")
        if join_op.remove:
            remove_cols = set(join_op.remove) & set(joined.columns)
            joined = joined.drop(columns=remove_cols)
    return joined


if __name__ == "__main__":

    tables = {
        "A": pd.DataFrame({"b": [1, 2, 3, 33], "c": [4, 5, 6, 66]}),
        "B": pd.DataFrame(
            {
                "b": [1, 2, 3],
                "a": [4, 5, 6],
                "d": [7, 8, 9],
                "e": [10, 11, 12],
                "f": [13, 14, 15],
            }
        ),
        "C": pd.DataFrame({"f": [13, 14, 15], "g": [4, 5, 6]}),
        "D": pd.DataFrame(
            {"d": [7, 8, 77], "e": [10, 11, 77], "h": [7, 8, 9], "i": [1, 2, 3]}
        ),
        "E": pd.DataFrame({"i": [1, 2, 3], "j": [4, 5, 6]}),
    }
    target_sub_set = {"b", "g", "j"}

    leaf_edges = get_leaf_edges(tables, target_sub_set)
    print(leaf_edges)
    join_sequence = generate_join_sequence(tables, leaf_edges, target_sub_set)
    print(join_sequence)
    join_result = compute_join_resolution(join_sequence, tables)
    print(join_result)

    # expected_join_resolution = [
    #     'B',
    #     Join('C', remove=['a', 'f']),
    #     Join('D', remove=['d', 'e', 'h']),
    #     Join('E', remove=['i'])
    # ]
    # expected_result = pd.DataFrame({
    #     'b': [1, 2],
    #     'g': [4, 5],
    #     'j': [4, 5]
    # })
