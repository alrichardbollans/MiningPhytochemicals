import os
import pickle
import time
from collections import defaultdict

import pandas as pd
import requests
from requests import HTTPError
from tqdm import tqdm

from data.get_compound_occurences import data_path, plantae_compounds_csv

api_endpoint = "https://api.core.ac.uk/v3/"

fulltext_dir = os.path.join(data_path, 'fulltexts')
pdf_dir = os.path.join(data_path, 'pdfs')


def extract_info(hit):
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


def get_results_for_doi(doi):
    if doi in request_dict_info:
        return request_dict_info[doi]

    time.sleep(1)
    headers = {"Authorization": "Bearer " + apikey}

    response = requests.get(f"{api_endpoint}search/works/?q=doi:{doi}", headers=headers)
    if not response.status_code == 200:
        # retry
        time.sleep(6)  # Rate limiting
        response = requests.get(f"{api_endpoint}search/works/?q=doi:{doi}", headers=headers)
    if response.status_code == 200:

        result = response.json()
        relevant_results = [c for c in result['results'] if c['doi'] and c['doi'].lower() == doi.lower()]
        out_dict = [extract_info(c) for c in relevant_results]

        request_dict_info[doi] = out_dict
        with  open(request_pkl_file, 'wb') as pfile:
            pickle.dump(request_dict_info, pfile)
        # print(request_dict_info)
        return out_dict

    elif response.status_code == 429:
        raise ValueError('Rate limit exceeded')
    else:
        raise HTTPError(f'Error getting results for {doi}. response code: {response.status_code}')


def sanitise_doi(doi):
    return doi.replace('/', '_')


def build_text_data(dois):
    for i in tqdm(range(len(dois))):
        doi = dois[i]
        text_out_file = os.path.join(fulltext_dir, sanitise_doi(doi) + '.txt')
        if os.path.exists(text_out_file):
            print(f'Skipping {doi} as it already exists')
        else:
            # print(f'Getting {doi}')
            try:
                r = get_results_for_doi(doi)
                fulltexts = [c['fullText'] for c in r if c['fullText']]

                if len(fulltexts) > 0:
                    ## Use largest fulltext just in case
                    sorted_fulltexts = sorted(fulltexts, key=lambda x: len(x), reverse=True)
                    with open(text_out_file, 'w') as outfile:
                        outfile.write(sorted_fulltexts[0])
            except HTTPError as e:
                print(f'Error: {e}')


def build_pdf_data(dois):
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
    compounds = pd.read_csv(plantae_compounds_csv)
    dois = compounds['refDOI'].unique().tolist()
    assert len(dois) == len(set([sanitise_doi(c) for c in dois]))
    build_text_data(dois)
    build_pdf_data(dois)


if __name__ == '__main__':
    with open('secrets.txt') as keyfile:
        apikey = keyfile.read()

    # Some pkls to store info about previous searches
    request_pkl_file = os.path.join(data_path, 'doi_request_dict.pkl')
    try:
        request_dict_info = pickle.load(open(request_pkl_file, 'rb'))
    except FileNotFoundError:
        request_dict_info = {}

    main()
