import pandas as pd
import requests
from bs4 import BeautifulSoup

def extract_wikipedia_tables(wikiurl):
    try:
        # Send an HTTP GET request to the Wikipedia URL
        response = requests.get(wikiurl)
        response.raise_for_status() 
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = soup.find_all('table', {'class': "wikitable"})

        if tables:
            df_list = []
            for table in tables:
                df = pd.read_html(str(table))
                if df:
                    df_list.append(pd.DataFrame(df[0]))
                else:
                    print("No data found in one of the tables.")
            
            if df_list:
                return df_list
            else:
                print("No tables found with class 'wikitable' containing data.")
        else:
            print(f"No tables with class 'wikitable' found on the page.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the HTTP request: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None

if __name__ == "__main__":
    wikiurl = "https://fr.wikipedia.org/wiki/Liste_des_communes_de_France_les_plus_peupl%C3%A9es"
    resulting_dfs = extract_wikipedia_tables(wikiurl)
    print(resulting_dfs[0].columns)
    if resulting_dfs is not None:
        for idx, df in enumerate(resulting_dfs):
            print(f"Table {idx + 1}:")
            print(df.head())
    else:
        print("Extraction failed.")
