"""To work with html"""

from typing import Callable, Union, Mapping, Optional, Sequence
import re
import io

import pandas as pd

DFLT_CHROME_WAIT = 3


def url_to_html_func(kind="requests") -> Callable:
    """Get a url_to_html function of a given kind."""

    # If kind is a tuple, the first element is the kind and the second element is the kwargs
    # to be used to parametrize the function
    # NOTE: For now, I'm just simply passing the kwargs to the place I think is most
    #  needed, but we can use a more sophisticated way to pass the kwargs to the right
    #  place using i2.call_forgivingly or i2.Sig.extract_args_and_kwargs or such
    if not isinstance(kind, str):
        kind_tuple = kind
        kind, kind_kwargs = kind_tuple
    else:
        kind_kwargs = {}

    url_to_html = None
    if kind == "requests":
        import requests  # pip install requests

        def url_to_html(url):
            r = requests.get(url, **kind_kwargs)
            if r.status_code != 200:
                print(
                    f"An error occured. Returning the response object for you to analyze: {r}"
                )
                return r
            return r.content

    elif kind == "chrome" or kind == "selenium":

        from selenium import webdriver  # pip install selenium
        from time import sleep

        dflt_wait = kind_kwargs.get("wait", DFLT_CHROME_WAIT)

        def url_to_html(url, wait=dflt_wait):
            b = webdriver.Chrome()
            b.get(url)
            if isinstance(wait, (int, float)):
                sleep(wait)
            html = b.page_source
            b.close()
            return html

    else:
        raise ValueError(f"Unknown url_to_html value: {url_to_html}")
    assert callable(url_to_html), "Couldn't make a url_to_html function"

    return url_to_html


get_tables_from_html = pd.read_html


TableFilter = Union[str, Sequence[str], Callable]


def _ensure_table_filter(filt: TableFilter) -> Callable:
    """
    Ensure that the filter is a callable that takes a dataframe and returns a boolean.

    If filt is a string, the column names must contain at least one column that matches filt regex.
    If filt is a list of strings, the column names must contain at least one column that matches
    the regexes in the filt list, for each regex in the list.
    If filt is a callable, it is used as is.

    Raises a ValueError if the filter type is unknown.

    Examples:

    >>> filt_func = _ensure_table_filter('foo')
    >>> bool(filt_func(pd.DataFrame({'foo': [1, 2]})))
    True
    >>> bool(filt_func(pd.DataFrame({'bar': [1, 2]})))
    False
    >>> filt_func = _ensure_table_filter(['foo', 'bar'])
    >>> bool(filt_func(pd.DataFrame({'football': [1, 2], 'baring': [3, 4]})))
    True
    >>> bool(filt_func(pd.DataFrame({'football': [1, 2], 'neither': [3, 4]})))
    False

    But if a same column name matches both regexes, it should return True:

    >>> bool(filt_func(pd.DataFrame({'foobar': [1, 2], 'huh': [3, 4]})))
    True

    """
    if filt is None:
        return lambda x: True
    if isinstance(filt, str):
        # The column names must contain at least one column that matches filt regex
        filt_regex = re.compile(filt)
        return lambda x: x.columns.str.contains(filt_regex).any()
    if isinstance(filt, Sequence):
        # The column names must contain at least one column that matches the
        # regexes in the filt list, for each regex in the list
        filt_regexes = [re.compile(f) for f in filt]
        return lambda x: all(x.columns.str.contains(f).any() for f in filt_regexes)
    if callable(filt):
        return filt
    raise ValueError(f"Unknown filter type: {filt}")


def get_tables_from_url(
    url,
    *,
    url_to_html: Union[Callable, str] = "requests",
    filt: Optional[TableFilter] = None,
    encoding: str = "utf-8",
    **tables_from_html_kwargs,
):
    """Get's a list of pandas dataframes from tables scraped from a url.
    Note that this will only work with static pages. If the html needs to be rendered dynamically,
    you'll have to get your needed html otherwise (like with selenium).

    >>> url = 'https://en.wikipedia.org/wiki/List_of_musical_instruments'
    >>> tables = get_tables_from_url(url)  # doctest: +SKIP

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
    - Download seleniumdriver here: https://chromedriver.chromium.org/
    - Uzip and put in a place that's on you PATH (run command `echo $PATH` for a list of those places)

    """
    if not callable(url_to_html):
        url_to_html = url_to_html_func(url_to_html)

    filt_func = _ensure_table_filter(filt)

    try:
        html = url_to_html(url)
        if isinstance(html, bytes):
            html = html.decode(encoding)
        tables = get_tables_from_html(io.StringIO(html), **tables_from_html_kwargs)
        return list(filter(filt_func, tables))
    except ValueError as e:
        if len(e.args) > 0:
            msg, *_ = e.args
            if "No tables found" in msg:
                return []
        raise


HTML_TEMPLATE1 = """
<html>
<head>
<style>
  h2 {
    text-align: center;
    font-family: Helvetica, Arial, sans-serif;
  }
  table { 
    margin-left: auto;
    margin-right: auto;
  }
  table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
  }
  th, td {
    padding: 5px;
    text-align: center;
    font-family: Helvetica, Arial, sans-serif;
    font-size: 90%;
  }
  table tbody tr:hover {
    background-color: #dddddd;
  }
  .wide {
    width: 90%; 
  }
</style>
</head>
<body>
"""

HTML_TEMPLATE2 = """
</body>
</html>
"""


def df_to_html(df, title=None):
    ht = ""
    if title is not None:
        ht += f"<h2> {title} </h2>\n"
    ht += df.to_html(classes="wide", escape=False)
    return ht


def df_store_to_html(df_store, sep="\n<br>\n"):
    ht = ""
    for k, df in df_store.items():
        title = re.match(r"[^\d]+", k).group(0)
        ht += df_to_html(df, title)
        ht += sep
    return ht


def dfs_to_html_pretty(dfs, title=None):
    """
    Write an entire dataframe to an HTML file
    with nice formatting.
    Thanks to @stackoverflowuser2010 for the
    pretty printer see https://stackoverflow.com/a/47723330/362951
    """

    if isinstance(dfs, pd.DataFrame):
        ht = df_to_html(dfs, title=title)
    elif isinstance(dfs, Mapping):
        ht = df_store_to_html(dfs)
    else:
        ht = df_store_to_html(dict(enumerate(dfs)))

    return HTML_TEMPLATE1 + ht + HTML_TEMPLATE2


def dfs_to_pdf_bytes(dfs, title=None):
    import weasyprint

    html = dfs_to_html_pretty(dfs, title)
    return weasyprint.HTML(string=html).write_pdf()
