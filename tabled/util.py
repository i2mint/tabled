"""
Utils
"""

from functools import partial
from typing import Mapping, KT, VT, Iterable, Callable
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
