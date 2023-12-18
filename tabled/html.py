"""To work with html"""

from typing import Callable, Union, Mapping
import re
import pandas as pd


def url_to_html_func(kind='requests') -> Callable:
    """Get a url_to_html function of a given kind."""
    url_to_html = None
    if kind == 'requests':
        import requests  # pip install requests

        def url_to_html(url):
            r = requests.get(url)
            if r.status_code != 200:
                print(
                    f'An error occured. Returning the response object for you to analyze: {r}'
                )
                return r
            return r.content

    elif kind == 'chrome' or kind == 'selenium':

        from selenium import webdriver  # pip install selenium
        from time import sleep

        def url_to_html(url, wait=2):
            b = webdriver.Chrome()
            b.get(url)
            if isinstance(wait, (int, float)):
                sleep(wait)
            html = b.page_source
            b.close()
            return html

    else:
        raise ValueError(f'Unknown url_to_html value: {url_to_html}')
    assert callable(url_to_html), "Couldn't make a url_to_html function"

    return url_to_html


get_tables_from_html = pd.read_html


def get_tables_from_url(
    url, *, url_to_html: Union[Callable, str] = 'requests', **tables_from_html_kwargs
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
    ```
        pip install selenium
        Download seleniumdriver here: https://chromedriver.chromium.org/
        Uzip and put in a place that's on you PATH (run command `echo $PATH` for a list of those places)
    ```
    """
    if not callable(url_to_html):
        url_to_html = url_to_html_func(url_to_html)
    try:
        return get_tables_from_html(url_to_html(url), **tables_from_html_kwargs)
    except ValueError as e:
        if len(e.args) > 0:
            msg, *_ = e.args
            if 'No tables found' in msg:
                return []
        raise


HTML_TEMPLATE1 = '''
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
'''

HTML_TEMPLATE2 = '''
</body>
</html>
'''


def df_to_html(df, title=None):
    ht = ''
    if title is not None:
        ht += f'<h2> {title} </h2>\n'
    ht += df.to_html(classes='wide', escape=False)
    return ht


def df_store_to_html(df_store, sep='\n<br>\n'):
    ht = ''
    for k, df in df_store.items():
        title = re.match(r'[^\d]+', k).group(0)
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
