"""Generated round-trip tests for pandas serializers (seeds x serializers).

Each serializer is a ``partial`` of a pandas ``to_*`` method, paired with the matching
``pd.read_*`` function. Every (seed, serializer) pair is written to a file whose name
carries the extension and the parametrization (``key=val`` naming), then read back.
A pair whose optional dependency is missing is skipped; a pair that fails otherwise is
a test failure. Lossless formats must also reproduce the seed exactly.
"""

from functools import partial
from itertools import product

import pandas as pd
import pytest

SEEDS = {
    "numeric": lambda: pd.DataFrame({"a": [1, 2, 3], "b": [0.5, 1.5, 2.5]}),
    "mixed": lambda: pd.DataFrame(
        {
            "i": [1, 2, 3],
            "f": [0.1, None, 0.3],
            "s": ["x", "y", "z"],
            "b": [True, False, True],
        }
    ),
    "dates": lambda: pd.DataFrame(
        {
            "t": pd.to_datetime(["2020-01-01", "2020-06-15", "2021-12-31"]),
            "v": [10, 20, 30],
        }
    ),
}

# name -> (writer(df, path), reader(path), lossless)
# ``name`` is "<ext>" or "<ext>,<key>=<val>,..." and is used in the file name.
SERIALIZERS = {
    "csv,index=False": (
        partial(pd.DataFrame.to_csv, index=False),
        pd.read_csv,
        False,
    ),
    "csv,sep=;,index=False": (
        partial(pd.DataFrame.to_csv, sep=";", index=False),
        partial(pd.read_csv, sep=";"),
        False,
    ),
    "json,orient=records": (
        partial(pd.DataFrame.to_json, orient="records", date_format="iso"),
        partial(pd.read_json, orient="records"),
        False,
    ),
    "pickle": (pd.DataFrame.to_pickle, pd.read_pickle, True),
    "parquet": (pd.DataFrame.to_parquet, pd.read_parquet, True),
    "feather": (pd.DataFrame.to_feather, pd.read_feather, True),
    "html": (
        partial(pd.DataFrame.to_html, index=False),
        lambda path: pd.read_html(path)[0],
        False,
    ),
}

_MISSING_DEP = (ImportError, ModuleNotFoundError)


@pytest.mark.parametrize(
    "seed_name, serializer_name", list(product(SEEDS, SERIALIZERS))
)
def test_serializer_round_trip(seed_name, serializer_name, tmp_path):
    write, read, lossless = SERIALIZERS[serializer_name]
    df = SEEDS[seed_name]()
    ext, *params = serializer_name.split(",")
    path = tmp_path / f"{seed_name},{','.join(params)}.{ext}".replace(",.", ".")

    try:
        write(df, path)
        back = read(path)
    except _MISSING_DEP as e:
        pytest.skip(f"optional dependency missing for {ext}: {e}")

    assert path.exists()
    assert list(back.columns) == list(df.columns)
    assert len(back) == len(df)
    if lossless:
        pd.testing.assert_frame_equal(back, df)
