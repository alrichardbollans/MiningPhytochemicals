import glob
import pickle
import random
import time
import os
from random import sample

import pandas as pd
import requests
import pathlib

from data.get_wikidata import data_path
from data.parse_refs import api_endpoint, sanitise_doi, build_text_data, _core_apikey


def get_random_fulltexts():
    time.sleep(2)
    headers = {"Authorization": "Bearer " + _core_apikey}
    response = requests.get(f"{api_endpoint}search/works/?q=_exists_:fullText&limit=500",
                            headers=headers)

    if response.status_code == 200:
        # Searching for fieldsofstudy directly in API doesn't seem to work
        # and doing a large query with post filtering is not working.
        # fields = ['botany', None]

        fulltext_dir = os.path.join(data_path, 'texts', f'random papers', 'fulltexts')

        pathlib.Path(fulltext_dir).mkdir(parents=True, exist_ok=True)
        result = response.json()
        results_with_fulltext = [c for c in result['results'] if
                                 ((c['fullText'] == c['fullText']) and (c['fullText'] != '') and (len(
                                     c['fullText']) > 10))]
        results_with_dois = [c for c in results_with_fulltext if
                             ((c['doi'] == c['doi']) and (c['doi'] != '') and (c['doi'] is not None))]
        counter = 0
        for instance in results_with_dois:
            if counter < 10:
                doi = instance['doi']
                text = instance['fullText']
                print(counter)
                text_out_file = os.path.join(fulltext_dir, sanitise_doi(doi) + '.txt')
                with open(text_out_file, 'w', encoding="utf-8") as outfile:
                    outfile.write(text)
                counter += 1

    else:
        print(response.status_code)


def get_med_plant_full_texts():
    top_10000 = pd.read_csv('medicinals_top_10000.csv')[['title', 'DOI', 'plant_species_binomials_unique_total']].dropna(
        subset=['DOI'])
    top_10000 = top_10000[top_10000['plant_species_binomials_unique_total'] > 0]
    selected_dois = random.sample(top_10000['DOI'].tolist(), 10)
    print(selected_dois)
    data = top_10000[top_10000['DOI'].isin(selected_dois)]
    print(data[['DOI', 'title']])
    fulltext_dir = os.path.join(data_path, 'texts', f'medplant papers', 'fulltexts')
    build_text_data(selected_dois, fulltext_dir)


def get_sanitised_dois_for_papers(dir_name: str):
    random_txt_dir = os.path.join(data_path, 'texts', dir_name, 'fulltexts')
    extension = '.txt'
    result = []
    for file in os.listdir(random_txt_dir):
        if file.endswith(extension):
            print(os.path.join(random_txt_dir, file))

            result.append(file[:-4])

    return random_txt_dir, result


def main():
    # get_random_fulltexts()
    get_med_plant_full_texts()


if __name__ == '__main__':
    main()
