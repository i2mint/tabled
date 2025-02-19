import pandas as pd
from tabled.base import DataframeKvReader


# Test cases
def test_DataframeKvReader():

    df = pd.DataFrame(
        {"A": [1, 2, 1], "B": [4, 5, 4], "C": [7, 8, 9], "D": [10, 11, 12]}
    )

    kv_readers = [
        DataframeKvReader(df, ["A", "B"], ["C", "D"]),
        DataframeKvReader(df.set_index(["A", "B"]), ["A", "B"], ["C", "D"]),
        DataframeKvReader(df.set_index("A"), ["A", "B"], ["C", "D"]),
    ]

    for kv_reader in kv_readers:
        # Accessing values
        key = (1, 4)
        expected_df = pd.DataFrame([{"C": 7, "D": 10}, {"C": 9, "D": 12}], index=[0, 2])
        assert (
            kv_reader[key]
            .reset_index(drop=True)
            .equals(expected_df.reset_index(drop=True))
        )
        assert list(kv_reader) == [(1, 4), (2, 5)]


import os
import shutil
import tempfile
import pandas as pd
import pytest
from tabled import DfFiles


def test_df_files_functionality():
    """Test DfFiles functionality matching the demo notebook"""

    # Test data dictionary - same as notebook
    misc_small_dicts = {
        "fantasy_tavern_menu": {
            "item": ["Dragon Ale", "Elf Bread", "Goblin Stew"],
            "price": [7.5, 3.0, 5.5],
            "is_alcoholic": [True, False, False],
            "servings_left": [12, 25, 8],
        },
        "alien_abduction_log": {
            "abductee_name": ["Bob", "Alice", "Zork"],
            "location": ["Kansas City", "Roswell", "Jupiter"],
            "duration_minutes": [15, 120, 30],
            "was_returned": [True, False, True],
        },
    }

    # Create temporary directory for testing
    rootdir = os.path.join(tempfile.gettempdir(), "tabled_df_files_test")
    if os.path.exists(rootdir):
        shutil.rmtree(rootdir)
    os.makedirs(rootdir)

    # Initialize DfFiles
    df_files = DfFiles(rootdir)

    # Verify it starts empty
    assert list(df_files) == []

    # Create DataFrames from test data
    fantasy_tavern_menu_df = pd.DataFrame(misc_small_dicts["fantasy_tavern_menu"])
    alien_abduction_log_df = pd.DataFrame(misc_small_dicts["alien_abduction_log"])

    # Save DataFrames using different formats
    df_files["fantasy_tavern_menu.csv"] = fantasy_tavern_menu_df
    df_files["alien_abduction_log.json"] = alien_abduction_log_df

    # Read data back and verify
    saved_df = df_files["fantasy_tavern_menu.csv"]
    pd.testing.assert_frame_equal(saved_df, fantasy_tavern_menu_df)

    # Test MutableMapping interface
    assert len(df_files) == 2
    assert sorted(list(df_files)) == [
        "alien_abduction_log.json",
        "fantasy_tavern_menu.csv",
    ]
    assert "fantasy_tavern_menu.csv" in df_files

    # Verify supported extensions
    encoder_extensions = list(df_files.extension_encoder_mapping)
    decoder_extensions = list(df_files.extension_decoder_mapping)

    # Test common formats
    extensions_supported_by_both = sorted(
        set(encoder_extensions) & set(decoder_extensions)
    )
    #      ... Reproduced here since not all the formats are actually working yet

    test_extensions = [
        "csv",
        "feather",  # requires `pip install pyarrow``
        "json",
        "orc",  # requires `pip install pyarrow`` or `pip install fastparquet`
        "parquet",  # requires `pip install pyarrow`` or `pip install fastparquet`
        "pkl",
        "tsv",
        # 'dta',  # TODO: fix
        # 'h5',  # TODO: fix
        # 'html',  # TODO: fix
        # 'sql',  # TODO: fix
        # 'xml',  # TODO: fix
    ]
    for ext in test_extensions:
        filename = f"test_file.{ext}"
        df_files[filename] = fantasy_tavern_menu_df
        df_loaded = df_files[filename]

        # Compare DataFrames ignoring index
        pd.testing.assert_frame_equal(
            fantasy_tavern_menu_df.reset_index(drop=True),
            df_loaded.reset_index(drop=True),
        )

    # Cleanup
    shutil.rmtree(rootdir)
