import time

import requests

from data.parse_refs import api_endpoint


def get_dois(field:str = None):

    time.sleep(2)
    headers = {"Authorization": "Bearer " + apikey}
    if field is not None:
        params = {
            'query': f'fieldOfStudy:"{field}"',
            '_exists_': 'fullText',
            "limit": 50
        }
    else:
        params = {
        }
    response = requests.get(f"{api_endpoint}search/works", headers=headers, params=params)

    if field is not None:
        fulltext_dir =   os.path.join(data_path, f'{field} papers','fulltexts')
    else:
        fulltext_dir =   os.path.join(data_path, f'random papers','fulltexts')

    if response.status_code == 200:

        result = response.json()
        for instance in result['results']:
            doi = instance['doi']
            text = instance['fullText']
            text_out_file = os.path.join(fulltext_dir, sanitise_doi(doi) + '.txt')
            with open(text_out_file, 'w') as outfile:
                outfile.write(text)


def main():
    botany_dois = get_dois('Botany')

if __name__ == '__main__':
    with open('secrets.txt') as keyfile:
        apikey = keyfile.read()

    main()