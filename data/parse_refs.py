import os
import pathlib
import pickle
import time
from collections import defaultdict

import pandas as pd
import requests
from requests import HTTPError
from tqdm import tqdm

from data.get_wikidata import data_path, wikidata_plantae_compounds_csv

api_endpoint = "https://api.core.ac.uk/v3/"

fulltext_dir = os.path.join(data_path, 'wikidata_fulltexts')
pdf_dir = os.path.join(data_path, 'pdfs')

# Some pkls to store info about previous searches
request_pkl_file = os.path.join(data_path, 'doi_request_dict.pkl')
try:
    with open(request_pkl_file, 'rb') as _pfile:
        _request_dict_info = pickle.load(_pfile)
except FileNotFoundError:
    _request_dict_info = {}

with open(os.path.join(data_path, 'secrets.txt')) as keyfile:
    _core_apikey = keyfile.read()


def extract_info(hit: dict) -> dict:
    """
    Extract specific information from a dictionary and organize it into a new dictionary.

    This function iterates over a set of predefined keys to extract their values from the
    input dictionary. If the key does not exist in the input dictionary, it skips without
    throwing an error. Additionally, it attempts to extract a nested value for the language
    key if present.

    :param hit: The input dictionary containing data to be extracted.
    :type hit: dict
    :return: A dictionary containing extracted key-value pairs. If a key is not found in
             the input dictionary, it is omitted. For the 'language' key, the nested
             'code' value is extracted if available.
    :rtype: dict
    """
    out_dict = defaultdict(None)
    for key in ['id', 'downloadUrl', 'doi', 'fullText', 'sourceFulltextUrls']:
        try:
            out_dict[key] = hit[key]
        except KeyError:
            pass
    try:
        out_dict['language'] = hit["language"]['code']
    except KeyError:
        pass
    return out_dict


def get_results_for_doi(doi: str) -> list[dict]:
    """
    Fetches results associated with a given DOI (Digital Object Identifier) by querying an external API.
    The function handles rate-limiting scenarios and retries requests if necessary. Results are cached locally
    for future lookups.

    :param doi: The Digital Object Identifier for which results are to be fetched.
    :type doi: str
    :return: A list of dictionaries containing extracted information for the given DOI.
    :rtype: list[dict]
    """

    if doi in _request_dict_info:
        return _request_dict_info[doi]
    # else:
    #     raise HTTPError(f'Error getting results for {doi}')

    time.sleep(2)
    headers = {"Authorization": "Bearer " + _core_apikey}

    # NOte this params method is commented out as it seems to be broken, although its following the example
    # here: https://api.core.ac.uk/docs/v3#tag/Works/operation/optionsCustomSearchWorks
    # params = {
    #     'q': f'doi:"{doi}"',
    # }
    # response = requests.get(f"{api_endpoint}search/works", headers=headers, params=params)
    response = requests.get(f"{api_endpoint}search/works/?q=doi:{doi}", headers=headers)

    if response.status_code == 429:
        # retry
        time.sleep(10)  # Rate limiting
        response = requests.get(f"{api_endpoint}search/works/?q=doi:{doi}", headers=headers)
    if response.status_code == 200:

        result = response.json()
        relevant_results = [c for c in result['results'] if c['doi'] and c['doi'].lower() == doi.lower()]
        out_dict = [extract_info(c) for c in relevant_results]

        _request_dict_info[doi] = out_dict
        with open(request_pkl_file, 'wb') as pfile:
            pickle.dump(_request_dict_info, pfile)
        # print(request_dict_info)
        return out_dict

    elif response.status_code == 429:
        raise ValueError('Rate limit exceeded')
    elif response.status_code == 500:
        raise HTTPError(f'Error getting results for {doi}. response code: {response.status_code}')
    else:
        raise ValueError(f'Something else is wrong. response code: {response.status_code}')


def sanitise_doi(doi: str) -> str:
    """
    Sanitises a given DOI (Digital Object Identifier) by replacing forward slashes
    ('/') with underscores ('_'). This ensures the DOI is formatted in a way
    suitable for systems or scenarios where forward slashes may be problematic.

    :param doi: A digital object identifier (DOI) string that may contain forward
        slashes ('/').
    :type doi: str
    :return: A modified version of the DOI, where all forward slashes are replaced
        with underscores.
    :rtype: str
    """
    return doi.replace('/', '_')


def build_text_data(dois: list[str], outfolder: str = fulltext_dir) -> None:
    """
    Processes a list of Digital Object Identifiers (DOIs) to retrieve full text data
    and saves them to text files. Skips DOIs for which the corresponding text file
    already exists. Utilizes the largest retrieved full text when multiple are available.

    :param dois: A list of DOIs to process
    :type dois: list[str]
    :return: None
    """
    pathlib.Path(outfolder).mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(dois))):
        doi = dois[i]
        text_out_file = os.path.join(outfolder, sanitise_doi(doi) + '.txt')
        if not os.path.exists(text_out_file):
            #     print(f'Skipping {doi} as it already exists')
            # else:
            # print(f'Getting {doi}')
            try:
                r = get_results_for_doi(doi)
                fulltexts = [c['fullText'] for c in r if c['fullText']]

                if len(fulltexts) > 0:
                    ## Use largest fulltext just in case
                    sorted_fulltexts = sorted(fulltexts, key=lambda x: len(x), reverse=True)
                    with open(text_out_file, 'w', encoding="utf-8") as outfile:
                        outfile.write(sorted_fulltexts[0])
            except HTTPError as e:
                print(f'Error: {e}')


def build_pdf_data(dois: list[str]) -> None:
    """
    Processes a list of DOIs to fetch and save PDF files associated with each DOI. The function checks if the
    PDF file for the DOI already exists, skips the download if so, or attempts to retrieve and save the PDF
    from a specified URL if available. Errors encountered during the fetch attempt are handled and printed.

    :param dois: A list of DOIs for which PDF files need to be fetched.
    :type dois: list[str]
    :return: This function does not return any value, but writes PDF files to the specified directory
        if fetched successfully.
    :rtype: None
    """
    for i in tqdm(range(len(dois))):
        doi = dois[i]
        pdf_out_file = os.path.join(pdf_dir, sanitise_doi(doi) + '.pdf')
        if os.path.exists(pdf_out_file):
            print(f'Skipping {doi} as it already exists')
        else:
            try:
                r = get_results_for_doi(doi)
                urls = [c['downloadUrl'] for c in r if c['downloadUrl'] and 'core.ac.uk' in c['downloadUrl']]

                if len(urls) > 0:
                    print(f'Downloading {doi} from {urls[0]}')
                    response = requests.get(urls[0])
                    with open(pdf_out_file, 'wb') as outfile:
                        outfile.write(response.content)
            except HTTPError as e:
                print(f'Error: {e}')


def main():
    compounds = pd.read_csv(wikidata_plantae_compounds_csv)
    dois = compounds['refDOI'].unique().tolist()
    assert len(dois) == len(set([sanitise_doi(c) for c in dois]))
    build_text_data(dois)


if __name__ == '__main__':
    main()
