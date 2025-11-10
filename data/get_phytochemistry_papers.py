import os
import pathlib
import random
import time

import pandas as pd
import requests

from data.get_wikidata import data_path
from data.parse_refs import api_endpoint, _core_apikey, sanitise_doi


def get_phytochem_fulltexts():
    time.sleep(2)
    headers = {"Authorization": "Bearer " + _core_apikey}
    response = requests.get(f'{api_endpoint}search/works/?q=(title:phytochemistry)&_exists_:fullText&limit=500',
                            headers=headers)

    if response.status_code == 200:
        # Searching for fieldsofstudy directly in API doesn't seem to work
        # and doing a large query with post filtering is not working.
        # fields = ['botany', None]

        fulltext_dir = os.path.join(data_path,'texts', f'phytochemistry papers', 'fulltexts')

        pathlib.Path(fulltext_dir).mkdir(parents=True, exist_ok=True)
        result = response.json()
        results_with_fulltext = [c for c in result['results'] if
                                 ((c['fullText'] is not None) and (c['fullText'] != '') and (len(
                                     c['fullText']) > 10))]
        results_with_dois = [c for c in results_with_fulltext if
                             ((c['doi'] == c['doi']) and (c['doi'] != '') and (c['doi'] is not None))]
        random.shuffle(results_with_dois)
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


def main():
    get_phytochem_fulltexts()
    top_10000 = pd.read_csv('medicinals_top_10000.csv')[['title', 'DOI', 'journals']].dropna(
        subset=['DOI'])
    journals = top_10000['journals'].dropna().unique().tolist()
    print([c for c in journals if 'phytochemistry' in c])


if __name__ == '__main__':
    main()
