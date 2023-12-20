"""
Utils
"""


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
            return {k: {kk: len(vv) for kk, vv in v.items()} for k, v in graph.items()}
    elif edge_labels is False:
        return {k: set(v) for k, v in graph.items()}

    raise ValueError(f"Invalid value for edge_labels: {edge_labels}")
