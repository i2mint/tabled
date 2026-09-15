# tabled.html

To work with html

### Functions

| [`df_store_to_html`](#tabled.html.df_store_to_html)(df_store[, sep])                | Render each dataframe in `df_store`, titled by its key's leading non-digit prefix, joined by `sep`.   |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| [`df_to_html`](#tabled.html.df_to_html)(df[, title])                          | Render `df` as an HTML table, with an optional `<h2>` title above it.                                 |
| [`dfs_to_html_pretty`](#tabled.html.dfs_to_html_pretty)(dfs[, title])                 | Write an entire dataframe to an HTML file with nice formatting.                                       |
| [`dfs_to_pdf_bytes`](#tabled.html.dfs_to_pdf_bytes)(dfs[, title])                   | Render `dfs` (a DataFrame, a mapping, or an iterable of DataFrames) to PDF bytes.                     |
| [`get_tables_from_url`](#tabled.html.get_tables_from_url)(url, \*[, url_to_html, ...]) | Get's a list of pandas dataframes from tables scraped from a url.                                     |
| [`url_to_html_func`](#tabled.html.url_to_html_func)([kind])                         | Get a url_to_html function of a given kind.                                                           |

### tabled.html.df_store_to_html(df_store, sep='\\n<br>\\n')

Render each dataframe in `df_store`, titled by its key’s leading non-digit prefix, joined by `sep`.

### tabled.html.df_to_html(df, title=None)

Render `df` as an HTML table, with an optional `<h2>` title above it.

### tabled.html.dfs_to_html_pretty(dfs, title=None)

Write an entire dataframe to an HTML file
with nice formatting.
Thanks to @stackoverflowuser2010 for the
pretty printer see [https://stackoverflow.com/a/47723330/362951](https://stackoverflow.com/a/47723330/362951)

### tabled.html.dfs_to_pdf_bytes(dfs, title=None)

Render `dfs` (a DataFrame, a mapping, or an iterable of DataFrames) to PDF bytes.

Requires the optional `weasyprint` dependency.

### tabled.html.get_tables_from_url(url, , url_to_html='requests', filt=None, encoding='utf-8', \*\*tables_from_html_kwargs)

Get’s a list of pandas dataframes from tables scraped from a url.
Note that this will only work with static pages. If the html needs to be rendered dynamically,
you’ll have to get your needed html otherwise (like with selenium).

```pycon
>>> url = 'https://en.wikipedia.org/wiki/List_of_musical_instruments'
>>> tables = get_tables_from_url(url)
```

If you install selenium and download a chromedriver,
you can even use your browser to render dynamic html.
Say, to get updated coronavirus stats without a need to figure out the API
(I mean, why have to figure out the language of an API, when someone already did that
for you in their webpage!!):

```python
url = 'https://www.worldometers.info/coronavirus/?utm_campaign=homeAdvegas1?'
tables = get_tables_from_url(url, url_to_html='chrome')  # doctest: +SKIP
```

To make selenium work:

- `pip install selenium`
- Download seleniumdriver here: [https://chromedriver.chromium.org/](https://chromedriver.chromium.org/)
- Uzip and put in a place that’s on you PATH (run command `echo $PATH` for a list of those places)

### tabled.html.url_to_html_func(kind='requests')

Get a url_to_html function of a given kind.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)
