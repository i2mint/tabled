
# tabled

A (key-value) data-object-layer to get (pandas) tables from a variety of sources with ease

To install:	```pip install tabled```

## SQLite Database Support

Tabled provides seamless integration with SQLite databases through `DfFiles`:

```python
from tabled import DfFiles

# Automatic SQLite detection - just pass the database file path
df_files = DfFiles('my_database.db')

# Access tables as DataFrames
customers = df_files['customers.parquet']  # Full filename
orders = df_files['orders']                # Clean table name (both work)

# List available tables
print(list(df_files.keys()))  # ['customers.parquet', 'orders.parquet', ...]

# Or use the explicit method
df_files = DfFiles.from_sqlite_file('my_database.db')
```

Under the hood, SQLite tables are exported to temporary Parquet files for efficient access, with automatic cleanup when the program exits.

### SQLite Export Tools

For more control over SQLite data extraction, use the `sqlite_tools` module:

```python
from tabled.sqlite_tools import export_sqlite_to_dataframes, export_sqlite_to_parquet

# Export to DataFrames
tables = export_sqlite_to_dataframes('database.db')
customers_df = tables['customers']

# Export to Parquet files
export_sqlite_to_parquet('database.db', 'output_directory/')
```

## Table Analysis and Diagnosis

The `dataframe_info` function provides flexible analysis of pandas DataFrames:

```python
from tabled.diagnose import dataframe_info, register_info_func
import pandas as pd

# Analyze a DataFrame
df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
info = dataframe_info(df)
print(info['shape'])  # (3, 2)
print(info['columns'])  # ['a', 'b']

# Extend with custom analysis functions
def memory_usage(df):
    return df.memory_usage(deep=True).sum()

register_info_func('memory', memory_usage)
info = dataframe_info(df)
print(info['memory'])  # Memory usage in bytes
```

The analysis is completely customizable - you can register new analysis functions or provide custom info function dictionaries to focus on specific aspects of your data.



# DfFiles

This section demonstrates how to use `DfFiles` to store and retrieve pandas DataFrames using various file formats.

## Setup

First, let's import required packages and define our test data:


```python
import os
import shutil
import tempfile

import pandas as pd
from tabled import DfFiles

# Test data dictionary
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
    }
}
```

## Creating Test Directory

We'll create a temporary directory for our files:


```python
def create_test_directory():
    # Create a directory for the test files
    rootdir = os.path.join(tempfile.gettempdir(), 'tabled_df_files_test')
    if os.path.exists(rootdir):
        shutil.rmtree(rootdir)
    os.makedirs(rootdir)
    print(f"Created directory at: {rootdir}")
    return rootdir

rootdir = create_test_directory()
print(f"Created directory at: {rootdir}")
```

    Created directory at: /var/folders/mc/c070wfh51kxd9lft8dl74q1r0000gn/T/tabled_df_files_test
    Created directory at: /var/folders/mc/c070wfh51kxd9lft8dl74q1r0000gn/T/tabled_df_files_test


## Initialize DfFiles

Create a new DfFiles instance pointing to our directory:


```python
df_files = DfFiles(rootdir)
```

Let's verify it starts empty:


```python
list(df_files)
```




    []



## Creating and Saving DataFrames

Let's create DataFrames from our test data:


```python
fantasy_tavern_menu_df = pd.DataFrame(misc_small_dicts['fantasy_tavern_menu'])
alien_abduction_log_df = pd.DataFrame(misc_small_dicts['alien_abduction_log'])

print("Fantasy Tavern Menu:")
display(fantasy_tavern_menu_df)
print("\nAlien Abduction Log:")
display(alien_abduction_log_df)
```

    Fantasy Tavern Menu:



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item</th>
      <th>price</th>
      <th>is_alcoholic</th>
      <th>servings_left</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dragon Ale</td>
      <td>7.5</td>
      <td>True</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Elf Bread</td>
      <td>3.0</td>
      <td>False</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Goblin Stew</td>
      <td>5.5</td>
      <td>False</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>


    
    Alien Abduction Log:



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>abductee_name</th>
      <th>location</th>
      <th>duration_minutes</th>
      <th>was_returned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Kansas City</td>
      <td>15</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alice</td>
      <td>Roswell</td>
      <td>120</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Zork</td>
      <td>Jupiter</td>
      <td>30</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


Now let's save these DataFrames using different formats:


```python
df_files['fantasy_tavern_menu.csv'] = fantasy_tavern_menu_df
df_files['alien_abduction_log.json'] = alien_abduction_log_df
```

## Reading Data Back

Let's verify we can read the data back correctly:


```python
saved_df = df_files['fantasy_tavern_menu.csv']
saved_df
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item</th>
      <th>price</th>
      <th>is_alcoholic</th>
      <th>servings_left</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dragon Ale</td>
      <td>7.5</td>
      <td>True</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Elf Bread</td>
      <td>3.0</td>
      <td>False</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Goblin Stew</td>
      <td>5.5</td>
      <td>False</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



## MutableMapping Interface

DfFiles implements the MutableMapping interface, making it behave like a dictionary.

Let's see how many files we have:


```python
len(df_files)
```




    2



List all available files:


```python
list(df_files)
```




    ['fantasy_tavern_menu.csv', 'alien_abduction_log.json']



Check if a file exists:


```python
'fantasy_tavern_menu.csv' in df_files
```




    True



## Supported File Extensions

Let's see what file formats DfFiles supports out of the box.

(**Note that some of these will require installing extra packages, which you'll realize if you get an ImportError**)


```python
print("Encoder supported extensions:")
list_of_encoder_supported_extensions = list(df_files.extension_encoder_mapping)
print(*list_of_encoder_supported_extensions, sep=', ')
```

    Encoder supported extensions:
    csv, txt, tsv, json, html, p, pickle, pkl, npy, parquet, zip, feather, h5, hdf5, stata, dta, sql, sqlite, gbq, xls, xlsx, xml, orc



```python
print("Decoder supported extensions:")
list_of_decoder_supported_extensions = list(df_files.extension_decoder_mapping)
print(*list_of_decoder_supported_extensions, sep=', ')
```

    Decoder supported extensions:
    csv, txt, tsv, parquet, json, html, p, pickle, pkl, xml, sql, sqlite, feather, stata, dta, sas, h5, hdf5, xls, xlsx, orc, sav


## Testing Different Extensions

Let's try saving and loading our test DataFrame in different formats:


```python
extensions_supported_by_encoder_and_decoder = (
    set(list_of_encoder_supported_extensions) & set(list_of_decoder_supported_extensions)
)
sorted(extensions_supported_by_encoder_and_decoder)
```


    ['csv',
     'dta',
     'feather',
     'h5',
     'hdf5',
     'html',
     'json',
     'orc',
     'p',
     'parquet',
     'pickle',
     'pkl',
     'sql',
     'sqlite',
     'stata',
     'tsv',
     'txt',
     'xls',
     'xlsx',
     'xml']




```python

```


```python
def test_extension(ext):
    filename = f'test_file.{ext}'
    try:
        df_files[filename] = fantasy_tavern_menu_df
        df_loaded = df_files[filename]
        # test the decoded df is the same as the one that was saved (round-trip test)
        # Note that we drop the index, since the index is not saved in the file by default for all codecs
        pd.testing.assert_frame_equal(
            fantasy_tavern_menu_df.reset_index(drop=True),
            df_loaded.reset_index(drop=True),
        )
        return True
    except Exception as e:
        return False


test_extensions = [
    'csv',
    'feather',
    'json',
    'orc',
    'parquet',
    'pkl',
    'tsv',  
    # 'dta',  # TODO: fix
    # 'h5',  # TODO: fix
    # 'html',  # TODO: fix
    # 'sql',  # TODO: fix
    # 'xml',  # TODO: fix
]

for ext in test_extensions:
    print("Testing extension:", ext)
    success = test_extension(ext)
    if success:
        print(f"\tExtension {ext}: ✓")
    else:
        print('\033[91m' + f"\tFix extension {ext}: ✗" + '\033[0m')
        
    # marker = '✓' if success else '\033[91m✗\033[0m'
    # print(f"\tExtension {ext}: {marker}")
```

    Testing extension: csv
    	Extension csv: ✓
    Testing extension: feather
    	Extension feather: ✓
    Testing extension: json
    	Extension json: ✓
    Testing extension: orc
    	Extension orc: ✓
    Testing extension: parquet
    	Extension parquet: ✓
    Testing extension: pkl
    	Extension pkl: ✓
    Testing extension: tsv
    	Extension tsv: ✓
    Testing extension: dta
    [91m	Fix extension dta: ✗[0m
    Testing extension: h5
    [91m	Fix extension h5: ✗[0m
    Testing extension: html
    [91m	Fix extension html: ✗[0m
    Testing extension: sql
    [91m	Fix extension sql: ✗[0m
    Testing extension: xml
    [91m	Fix extension xml: ✗[0m


