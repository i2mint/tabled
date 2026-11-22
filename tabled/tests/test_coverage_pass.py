"""Coverage-pass tests for ``tabled.base`` (KeyFuncReader, DataframeKvReader, DfFiles,
DfReader and helpers), ``tabled.html`` and the cheap error path of ``sqlite_tools``."""

import sqlite3

import pandas as pd
import pytest

from tabled.base import (
    DataframeKvReader,
    DfFiles,
    DfReader,
    KeyFuncReader,
    _is_sqlite_file,
    convert_collection_to_dataframe_if_possible,
    validate_fields,
)
from tabled.html import df_store_to_html, df_to_html, dfs_to_html_pretty


def test_key_func_reader():
    r = KeyFuncReader({"a": 1, "b": 2}, key=str.lower)
    assert r["A"] == 1 and "B" in r and "c" not in r
    assert list(r) == ["a", "b"] and len(r) == 2
    assert repr(r).startswith("KeyFuncReader(")
    with pytest.raises(KeyError):
        r["zzz"]


def test_convert_collection_to_dataframe_if_possible():
    df = pd.DataFrame({"a": [1]})
    assert convert_collection_to_dataframe_if_possible(df) is df
    assert convert_collection_to_dataframe_if_possible({"a": [1, 2]}).shape == (2, 1)
    assert convert_collection_to_dataframe_if_possible([{"a": 1}]).shape == (1, 1)
    assert convert_collection_to_dataframe_if_possible(
        pd.Series([1, 2], name="s")
    ).shape == (2, 1)
    assert convert_collection_to_dataframe_if_possible(
        pd.Index([1, 2], name="i")
    ).shape == (2, 1)
    assert convert_collection_to_dataframe_if_possible("text") == "text"


def test_validate_fields():
    df = pd.DataFrame({"a": [1], "b": [2]}).set_index("a")
    validate_fields(df, ["a"], ["b"])
    with pytest.raises(ValueError, match="Field z"):
        validate_fields(df, ["z"], ["b"])
    with pytest.raises(ValueError, match="Column q"):
        validate_fields(df, ["a"], ["q"])


@pytest.fixture
def kv():
    df = pd.DataFrame({"A": [1, 2, 1], "B": [4, 5, 4], "C": [7, 8, 9]})
    return DataframeKvReader(df, ["A", "B"], ["C"])


def test_dataframe_kv_reader_mapping_interface(kv):
    assert list(kv) == [(1, 4), (2, 5)]
    assert len(kv) == 2
    assert (1, 4) in kv and (9, 9) not in kv
    assert kv[(1, 4)]["C"].tolist() == [7, 9]
    assert kv[(2, 5)]["C"].tolist() == [8]  # a single row stays a frame
    assert "key_fields=['A', 'B']" in repr(kv)
    with pytest.raises(KeyError, match="not found"):
        kv[(9, 9)]


def test_dataframe_kv_reader_scalar_key_and_defaults():
    df = pd.DataFrame({"A": [1, 2], "C": [3, 4]})
    r = DataframeKvReader(df, "A".split(), None)
    assert r[1].shape[0] == 1
    assert r.value_columns == ["A", "C"]


def test_df_files_roundtrip_and_reader(tmp_path):
    store = DfFiles(str(tmp_path))
    df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    store["t.csv"] = df
    store["u.json"] = df
    assert sorted(store) == ["t.csv", "u.json"]
    assert store["t.csv"]["y"].tolist() == ["a", "b"]
    # raw bytes are written as-is by default, and refused when not allowed
    store["raw.csv"] = b"x,y\n1,2\n"
    assert store["raw.csv"].shape == (1, 2)
    del store["raw.csv"]
    with pytest.raises(ValueError):
        DfFiles(str(tmp_path), allow_writing_bytes=False)["raw.csv"] = b"x,y\n1,2\n"
    reader = DfReader(str(tmp_path))
    assert reader["t.csv"].shape == (2, 2)
    with pytest.raises(NotImplementedError):
        reader["t.csv"] = df
    with pytest.raises(NotImplementedError):
        del reader["t.csv"]
    del store["u.json"]
    assert list(store) == ["t.csv"]


def test_is_sqlite_file(tmp_path):
    assert _is_sqlite_file("anything.sqlite3")
    db = tmp_path / "noext"
    with sqlite3.connect(db) as con:
        con.execute("create table t(a)")
    con.close()
    assert _is_sqlite_file(db)
    other = tmp_path / "plain"
    other.write_text("hello")
    assert not _is_sqlite_file(other)
    assert not _is_sqlite_file(tmp_path / "missing")


def test_html_helpers():
    df = pd.DataFrame({"a": ["<b>"]})
    assert "<h2> T </h2>" in df_to_html(df, "T") and "<b>" in df_to_html(df)
    assert "<h2>" not in df_to_html(df)
    html = df_store_to_html({"table1": df, "other2": df}, sep="|SEP|")
    assert "<h2> table </h2>" in html and "<h2> other </h2>" in html
    assert html.count("|SEP|") == 2
    for dfs in (df, {"k1": df}):
        page = dfs_to_html_pretty(dfs)
        assert "<table" in page and page.lstrip().lower().startswith(
            ("<!doctype", "<html", "<")
        )


def test_dfs_to_html_pretty_list_of_frames():
    page = dfs_to_html_pretty([pd.DataFrame({"a": [1]})])
    assert "<table" in page and "<h2> 0 </h2>" in page


def test_df_store_to_html_with_all_digit_string_keys():
    html = df_store_to_html({"2024": pd.DataFrame({"a": [1]})})
    assert "<h2> 2024 </h2>" in html


def test_sqlite_export_missing_file(tmp_path):
    pytest.importorskip("duckdb")
    from tabled.sqlite_tools import export_sqlite_to_dataframes

    with pytest.raises(FileNotFoundError):
        export_sqlite_to_dataframes(tmp_path / "nope.db")
