import os
import pathlib
import pickle
import time

import pandas as pd
import requests
from tqdm import tqdm
from wcvpy.wcvp_download import get_distributions_for_accepted_taxa, get_all_taxa
from wcvpy.wcvp_name_matching import get_genus_from_full_name

from data.get_knapsack_data import knapsack_plantae_compounds_csv
from data.get_wikidata import data_path, wikidata_plantae_compounds_csv, WCVP_VERSION
from data.parse_refs import api_endpoint, _core_apikey, sanitise_doi


def get_species_to_collect():
    wikidata = pd.read_csv(wikidata_plantae_compounds_csv, index_col=0)
    knapsack = pd.read_csv(knapsack_plantae_compounds_csv, index_col=0)
    all_data = pd.concat([wikidata, knapsack])
    all_taxa = get_all_taxa(ranks=['Species'], accepted=True, version=WCVP_VERSION)
    dists = get_distributions_for_accepted_taxa(all_taxa.drop_duplicates(subset=['accepted_species'], keep='first'), 'accepted_species',
                                                include_extinct=True,
                                                wcvp_version=WCVP_VERSION)
    print(dists.shape)
    dists = dists[~dists['accepted_species'].isin(all_data['accepted_species'].values)]
    print(dists.shape)
    dists = dists.dropna(subset=['native_tdwg3_codes'])
    dists = dists[dists['native_tdwg3_codes'].apply(lambda x: True if 'CLM' in x
    else False)]

    dists[['accepted_species', 'native_tdwg3_codes']].to_csv(os.path.join('colombian species not in datasets', 'species.csv'))
    dists.describe(include='all').to_csv(os.path.join('colombian species not in datasets', 'species_summary.csv'))

    return dists['accepted_species'].tolist()


def do_query(genus, scrollID=None):
    time.sleep(2)
    headers = {"Authorization": "Bearer " + _core_apikey}
    limit = 600
    # Expand search a bit due to lack of full texts
    search_string = f'((title:phytochemistry OR title:phytochemical OR title:metabolomics) AND (title:{genus}))&_exists_:fullText'
    if scrollID is None:
        response = requests.get(
            f'{api_endpoint}search/works/?q={search_string}&limit={str(limit)}&scroll=true',
            headers=headers)
    else:
        response = requests.get(
            f'{api_endpoint}search/works/?q=(title:phytochemistry)&_exists_:fullText&limit={str(limit)}&scrollId={scrollID}',
            headers=headers)
    if response.status_code == 429:
        raise Exception("Rate limit exceeded")
    if response.status_code != 200:
        return None
    result = response.json()
    results_with_fulltext = [c for c in result['results'] if
                             ((c['fullText'] is not None) and (c['fullText'] != '') and (len(
                                 c['fullText']) > 10))]
    results_with_dois = [c for c in results_with_fulltext if
                         ((c['doi'] == c['doi']) and (c['doi'] != '') and (c['doi'] is not None))]

    genus_species_to_collect = [c for c in species_to_collect if genus in c]
    results_for_species = [c for c in results_with_dois if any(sp.lower() in c['title'].lower() for sp in genus_species_to_collect)]

    sanitised_dois = []
    if len(results_for_species) > 0:

        for instance in results_with_dois:
            for sp in genus_species_to_collect:
                if sp.lower() in instance['title'].lower():
                    fulltext_dir = os.path.join(data_path, 'texts', f'colombian papers', 'fulltexts', sp)
                    pathlib.Path(fulltext_dir).mkdir(parents=True, exist_ok=True)
                    print(f'success for {sp}')

                    doi = instance['doi']
                    text = instance['fullText']
                    if sanitise_doi(doi) in sanitised_dois:
                        print(f'collision for {doi}: {sp}')
                    sanitised_dois.append(sanitise_doi(doi))
                    text_out_file = os.path.join(fulltext_dir, sanitise_doi(doi) + '.txt')
                    with open(text_out_file, 'w', encoding="utf-8") as outfile:
                        outfile.write(text)
    return result


def get_all_species_fulltexts(genus):
    scrollId = None
    while True:

        result = do_query(genus, scrollID=scrollId)
        if result is None:
            break
        scrollId = result["scrollId"]
        totalhits = result["totalHits"]
        result_size = len(result["results"])
        # print(f"scrollId: {scrollId}, totalHits: {totalhits}, result_size: {result_size}")
        if result_size == 0:
            break


def iterate_over_all_missing_species():
    search_pkl = os.path.join('colombian species not in datasets', 'tried_species.pkl')
    try:
        with open(search_pkl, 'rb') as _pfile:
            tried_genera = pickle.load(_pfile)
    except FileNotFoundError:
        tried_genera = []
    print(f'already tried {len(tried_genera)} genera')
    for i in tqdm(range(len(genera_to_search))):
        genus = genera_to_search[i]
        if genus not in tried_genera:
            get_all_species_fulltexts(genus)
            tried_genera.append(genus)
            with open(search_pkl, 'wb') as pfile:
                pickle.dump(tried_genera, pfile)


def get_sanitised_dois_for_colombian_papers():
    _txt_dir = os.path.join(data_path, 'texts', 'colombian papers', 'fulltexts')
    result = {}
    folders = os.listdir(_txt_dir)
    for sp in folders:
        sp_dir = os.path.join(_txt_dir, sp)
        for file in os.listdir(sp_dir):
            if file.endswith('.txt'):
                result[file[:-4]] = sp_dir

    return result


def main():
    # get_species_to_collect()
    # Check
    wikidata = pd.read_csv(wikidata_plantae_compounds_csv, index_col=0)
    knapsack = pd.read_csv(knapsack_plantae_compounds_csv, index_col=0)
    all_data = pd.concat([wikidata, knapsack])
    assert len(set(all_data['accepted_species'].tolist()).intersection(set(species_to_collect))) == 0

    iterate_over_all_missing_species()


if __name__ == '__main__':
    species_to_collect = pd.read_csv(os.path.join('colombian species not in datasets', 'species.csv'), index_col=0)['accepted_species'].tolist()
    genera_to_search = list(set([get_genus_from_full_name(sp_) for sp_ in species_to_collect]))
    main()
