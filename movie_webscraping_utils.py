# Import modules
import urllib
from bs4 import BeautifulSoup
import os
import pandas as pd

def is_absolute_url(url):
    '''
    Is url an absolute URL?
    '''
    if url == "":
        return False
    return urllib.parse.urlparse(url).netloc != ""

def read_request(url, headers={}):
    '''
    Return data from request object.  Returns result or "" if the read
    fails.
    
    Inputs:
        url: must be an absolute URL
        headers: dict
            HTTP header that can be used in an HTTP request to provide
            information about the request contex. Default value is an empty
            dictionary.

    Outputs:
        data from request object or ""
    '''
    if is_absolute_url(url):
        try:
            req = urllib.request.Request(url, headers=headers)
            response = urllib.request.urlopen(req)
            return response.read()
        except Exception:
            print("Read failed: " + url)
            return ""

def convert_if_relative_url(current_url, new_url):
    '''
    Attempt to determine whether new_url is a relative URL and if so,
    use current_url to determine the path and create a new absolute
    URL. Will add the protocol, if that is all that is missing.
    '''
    if new_url == "" or not is_absolute_url(current_url):
        return None

    if is_absolute_url(new_url):
        return new_url

    parsed_url = urllib.parse.urlparse(new_url)
    path_parts = parsed_url.path.split("/")

    if len(path_parts) == 0:
        return None

    ext = path_parts[0][-4:]
    if ext in [".edu", ".org", ".com", ".net"]:
        return "http://" + new_url
    elif new_url[:3] == "www":
        return "http://" + new_path
    else:
        return urllib.parse.urljoin(current_url, new_url)

def extract_wiki_plot(wiki_url):
    '''
    Request the wiki_url page, parse it and return plot section of 
    the movie webpage.
    '''
    movie_html_wiki = read_request(wiki_url)
    movie_soup_wiki = BeautifulSoup(movie_html_wiki, 'html.parser')
    # Extracting the plot summary
    wiki_plot = ''
    tag  = movie_soup_wiki.select_one('#Plot').find_parent('h2')
    while tag.name != 'p':
        tag = tag.find_next_sibling()
    while tag.name == 'p':
        wiki_plot += tag.text.replace('\n', ' ')
        tag = tag.find_next_sibling()
    return wiki_plot

def add_data_to_df_field(df, row, column, data):
    '''
    Add data to the dataframe field specified by the row and column provided
    if the field is empty. At the end, display the entire row containing the
    field in question.
    '''
    if df.loc[row, column].isnull().any() == True:
        df.loc[row, column] = data
    display(df[row])

def update_csv_file(name, row_data_list, cols, folder_path = './data/'):
    '''
    Update .csv file with the data for each new row that is being added.

    Inputs:
    -------
    name: string
        File name. If the file does not already exist creates a
        new file. Otherwise appends new data to the current .csv file.
    row_data_list: list
        Data that will populate a new row in existing .csv file.
    cols: list
        Names of columns within the .csv file.
        Has to be of the same length as row_data.
    folder_path: string, optional
        A path to the folder where the .csv file should be saved.
        Default value: './data/'
    '''
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        
    df_path = os.path.join(folder_path, name)
    
    if not os.path.isfile(df_path):
        df = pd.DataFrame(columns=cols)
        df.to_csv(df_path, index=False)
        print(name + ' file does not exist. Creating new file!')

    data_df = pd.read_csv(df_path)
    df = pd.DataFrame(row_data_list, columns=cols)
    data_df = data_df.append(df, ignore_index=True)
    data_df.to_csv(df_path, index=False)