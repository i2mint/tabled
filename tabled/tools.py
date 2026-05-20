"""Various high-level tools using tabled"""

from typing import Callable
from collections.abc import Mapping, Iterable
from tabled.diagnose import dataframe_info, DFLT_INFO_FUNCS
from tabled.base import DfFiles


def diagnose_table_collection(
    tables,
    *,
    info_funcs: dict[str, Callable] = DFLT_INFO_FUNCS,
    egress: Callable = dict,
):
    """Diagnose a collection of tables and return diagnostic information.

    Args:
        tables: Collection of tables - can be:
            - A mapping from keys to DataFrames
            - A non-mapping, non-string iterable of DataFrames (will use enumerate for keys)
            - A string URI to create DfFiles mapping
        info_funcs: Dictionary of info functions to apply
        egress: Function to process the generator of (table_key, info_dict) pairs

    Returns:
        Result of egress applied to the info generator

    Examples:
        >>> import pandas as pd

        # Mapping case
        >>> tables_dict = {'table1': pd.DataFrame({'a': [1, 2], 'b': [3, 4]})}
        >>> result = diagnose_table_collection(tables_dict)
        >>> 'table1' in result
        True

        # Iterable case
        >>> df1 = pd.DataFrame({'a': [1, 2]})
        >>> df2 = pd.DataFrame({'b': [3, 4]})
        >>> result = diagnose_table_collection([df1, df2])
        >>> 0 in result and 1 in result
        True
    """

    # Handle different input types
    if isinstance(tables, str):
        # String case: treat as URI and create DfFiles mapping
        tables_uri = tables
        tables = DfFiles(tables_uri)
    elif not isinstance(tables, Mapping):
        # Non-mapping iterable case: use enumerate for keys
        if isinstance(tables, Iterable):
            tables = dict(enumerate(tables))
        else:
            raise TypeError("tables must be a mapping, iterable, or string URI")

    def info_dicts_gen():
        for table_key, df in tables.items():
            info_dict = dataframe_info(df, info_funcs=info_funcs)
            yield table_key, info_dict

    return egress(info_dicts_gen())
