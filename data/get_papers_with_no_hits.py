import glob
import time
import os
from random import sample

import requests
import pathlib

from data.get_compound_occurences import data_path
from data.parse_refs import api_endpoint, sanitise_doi


def get_random_fulltexts():
    time.sleep(2)
    headers = {"Authorization": "Bearer " + apikey}
    response = requests.get(f"{api_endpoint}search/works/?q=_exists_:fullText&limit=500",
                            headers=headers)

    if response.status_code == 200:
        fields = [None]
        # Searching for fieldsofstudy directly in API doesn't seem to work
        # and doing a large query with post filtering is not working.
        # fields = ['botany', None]
        for field in fields:
            if field is not None:
                fulltext_dir = os.path.join(data_path, f'{field} papers', 'fulltexts')
            else:
                fulltext_dir = os.path.join(data_path, f'random papers', 'fulltexts')

            pathlib.Path(fulltext_dir).mkdir(parents=True, exist_ok=True)
            result = response.json()
            results_with_fulltext = [c for c in result['results'] if
                                     ((c['fullText'] == c['fullText']) and (c['fullText'] != '') and (len(
                                         c['fullText']) > 10))]
            results_with_dois = [c for c in results_with_fulltext if
                                 ((c['doi'] == c['doi']) and (c['doi'] != '') and (c['doi'] is not None))]
            relevant_results = [c for c in results_with_dois if ((field is None) or (c['fieldOfStudy'] == field))]

            counter = 0
            for instance in relevant_results:
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


def get_sanitised_dois_for_random_papers():
    random_txt_dir = os.path.join(data_path, f'random papers', 'fulltexts')
    extension = 'txt'
    os.chdir(random_txt_dir)
    result = glob.glob('*.{}'.format(extension))

    result = [c[:-4] for c in result[:]]

    return random_txt_dir, result


def main():
    get_random_fulltexts()


if __name__ == '__main__':
    with open('secrets.txt') as keyfile:
        apikey = keyfile.read()

    main()
