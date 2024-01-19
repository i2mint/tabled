import pandas as pd
import pytest
from join_tables import *


@pytest.fixture
def simple_tables():
    return {
        'A': pd.DataFrame({'x': [1, 2], 'y': [3, 4]}),
        'B': pd.DataFrame({'y': [3, 4], 'z': [5, 6]}),
    }


@pytest.fixture
def complex_tables():
    return {
        'A': pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}),
        'B': pd.DataFrame({'b': [3, 4], 'd': [7, 8]}),
        'C': pd.DataFrame({'d': [7, 8], 'e': [9, 10], 'f': [11, 12]}),
        'D': pd.DataFrame({'f': [11, 12], 'g': [13, 14]}),
    }


@pytest.fixture
def tables():
    return {
        'A': pd.DataFrame({'b': [1, 2, 3, 33], 'c': [4, 5, 6, 66]}),
        'B': pd.DataFrame(
            {
                'b': [1, 2, 3],
                'a': [4, 5, 6],
                'd': [7, 8, 9],
                'e': [10, 11, 12],
                'f': [13, 14, 15],
            }
        ),
        'C': pd.DataFrame({'f': [13, 14, 15], 'g': [4, 5, 6]}),
        'D': pd.DataFrame(
            {'d': [7, 8, 77], 'e': [10, 11, 77], 'h': [7, 8, 9], 'i': [1, 2, 3]}
        ),
        'E': pd.DataFrame({'i': [1, 2, 3], 'j': [4, 5, 6]}),
    }


# Test for the simple case
def test_minimum_covering_tree_simple(simple_tables):
    target_subset = ['y', 'z']
    edges = minimum_covering_tree(simple_tables, target_subset)
    assert set(edges) == {
        ('A', 'B')
    }, f"Expected edges: {set([('A', 'B')])}, but got: {set(edges)}"


def test_generate_join_sequence_simple(simple_tables):
    target_subset = {'y', 'z'}
    leaf_edges = get_leaf_edges(simple_tables, target_subset)
    join_sequence = generate_join_sequence(simple_tables, leaf_edges, target_subset)
    expected_sequence = ['A', Join('B', remove=['x'])]
    assert (
        join_sequence == expected_sequence
    ), f'Expected join sequence: {expected_sequence}, but got: {join_sequence}'


def test_compute_join_resolution_simple(simple_tables):
    target_subset = {'y', 'z'}
    leaf_edges = get_leaf_edges(simple_tables, target_subset)
    join_sequence = generate_join_sequence(simple_tables, leaf_edges, target_subset)
    expected_result = pd.DataFrame({'y': [3, 4], 'z': [5, 6]})
    result = compute_join_resolution(join_sequence, simple_tables)
    pd.testing.assert_frame_equal(result, expected_result)


# Test for the complex case
# def test_minimum_covering_tree_complex(complex_tables):
#     target_subset = ['a', 'e', 'g']
#     edges = minimum_covering_tree(complex_tables, target_subset)
#     expected_edges = {('A', 'B'), ('B', 'C'), ('C', 'D')}
#     assert set(edges) == expected_edges, f"Expected edges: {expected_edges}, but got: {set(edges)}"

# def test_generate_join_sequence_complex(complex_tables):
#     target_subset = {'a', 'e', 'g'}
#     leaf_edges = get_leaf_edges(complex_tables, target_subset)
#     join_sequence = generate_join_sequence(complex_tables, leaf_edges, target_subset)
#     expected_sequence = ['A', Join('B', remove=['c']), Join('C', remove=['b']), Join('D', remove=['d', 'f'])]
#     print(join_sequence)
#     assert join_sequence == expected_sequence, f"Expected join sequence: {expected_sequence}, but got: {join_sequence}"

# def test_compute_join_resolution_complex(complex_tables):
#     target_subset = {'a', 'e', 'g'}
#     leaf_edges = get_leaf_edges(complex_tables,target_subset)
#     join_sequence = generate_join_sequence(complex_tables, leaf_edges, target_subset)
#     expected_result = pd.DataFrame({'a': [1, 2], 'e': [9, 10], 'g': [13, 14]})
#     result = compute_join_resolution(join_sequence, complex_tables)
#     pd.testing.assert_frame_equal(result, expected_result)

# Test for tables
def test_minimum_covering_tree(tables):
    target_subset = {'b', 'g', 'j'}
    edges = minimum_covering_tree(tables, target_subset)
    expected_edges = {('A', 'B'), ('B', 'C'), ('B', 'D'), ('D', 'E')}
    assert (
        set(edges) == expected_edges
    ), f'Expected edges: {expected_edges}, but got: {set(edges)}'


def test_generate_join_sequence(tables):
    target_subset = {'b', 'g', 'j'}
    leaf_edges = get_leaf_edges(tables, target_subset)
    join_sequence = generate_join_sequence(tables, leaf_edges, target_subset)
    expected_sequence = [
        'B',
        Join('C', remove=['f', 'a']),
        Join('D', remove=['e', 'd', 'h']),
        Join('E', remove=['i']),
    ]
    assert (
        join_sequence == expected_sequence
    ), f'Expected join sequence: {expected_sequence}, but got: {join_sequence}'


def test_compute_join_resolution(tables):
    target_subset = {'b', 'g', 'j'}
    leaf_edges = get_leaf_edges(tables, target_subset)
    join_sequence = generate_join_sequence(tables, leaf_edges, target_subset)
    expected_result = pd.DataFrame({'b': [1, 2], 'g': [4, 5], 'j': [4, 5]})
    result = compute_join_resolution(join_sequence, tables)
    pd.testing.assert_frame_equal(result, expected_result)


from tabled.multi import execute_commands


def test_execute_commands_simply():
    from tabled.multi import Join, Remove, Load

    # ---------------------------------------------
    # First silly test

    silly_interpreter_map = {
        Join: lambda scope, command: f'Joining {command.table_key}',
        Remove: lambda scope, command: f'Removing {command.fields}',
    }

    g = execute_commands(
        [Join('asdf'), Remove('apple')], scope={}, interpreter_map=silly_interpreter_map
    )
    assert list(g) == ['Joining asdf', 'Removing apple']

    # ---------------------------------------------
    # The real case

    table1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
    table2 = pd.DataFrame({'ID': [2, 3, 4], 'Age': [25, 30, 22]})
    table3 = pd.DataFrame({'ID': [1, 2, 3, 4], 'Salary': [50000, 60000, 70000, 55000]})

    tables = {'table1': table1, 'table2': table2, 'table3': table3}

    commands = [Load('table1'), Remove(['Name']), Join('table3')]

    scope = tables
    extra_scope = dict()

    from tabled.multi import execute_table_commands

    it = execute_table_commands(commands, tables, extra_scope=extra_scope)

    def are_equal(a, b):
        if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
            return (a == b).all().all()
        else:
            return a == b

    next(it)
    assert are_equal(extra_scope['cumul'], scope['table1'])
    next(it)
    assert are_equal(extra_scope['cumul'], pd.DataFrame({'ID': [1, 2, 3]}))
    next(it)
    assert list(extra_scope['cumul']) == ['ID', 'Salary']
