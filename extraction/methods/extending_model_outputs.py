import os
import pickle
import urllib

import cirpy
import pandas as pd
from phytochempy.compound_properties import simplify_inchi_key
from wcvpy.wcvp_download import get_all_taxa
from wcvpy.wcvp_name_matching import get_accepted_info_from_names_in_column, get_genus_from_full_name

from data.get_compound_occurences import data_path
from extraction.methods.structured_output_schema import TaxaData, Taxon

# _wcvp_taxa = get_all_taxa()
def add_accepted_info(deepseek_output: TaxaData):
    deepseek_names = pd.DataFrame([c.scientific_name for c in deepseek_output.taxa], columns=['scientific_name'])
    acc_deepseek_names = get_accepted_info_from_names_in_column(deepseek_names, 'scientific_name', all_taxa = _wcvp_taxa)
    acc_deepseek_names = acc_deepseek_names.set_index('scientific_name')
    for taxon in deepseek_output.taxa:
        taxon.accepted_name = acc_deepseek_names.loc[taxon.scientific_name, 'accepted_name']
        taxon.accepted_species = acc_deepseek_names.loc[taxon.scientific_name, 'accepted_species']
        taxon.accepted_genus = get_genus_from_full_name(taxon.accepted_species)


def resolve_name_to_inchi(name: str):
    """

    """
    _inchi_translation_cache = os.path.join(data_path, 'inchi_translation_cache.pkl')
    try:
        pkled_result = pickle.load(open(_inchi_translation_cache, 'rb'))
    except FileNotFoundError:
        pkled_result = {}
    if name not in pkled_result:
        out = None
        if name == name and name != '':

            try:
                inch = cirpy.resolve(name, 'stdinchikey')
                if inch is not None:
                    out = inch.replace('InChIKey=', '')
            except (urllib.error.HTTPError, urllib.error.URLError):
                out = None
                print(f'WARNING: cas id not resolved: {name}')
        pkled_result[name] = out
    with  open(_inchi_translation_cache, 'wb') as pfile:
        pickle.dump(pkled_result,pfile)
    return pkled_result[name]


def add_inchi_keys(deepseek_output: TaxaData):
    for taxon in deepseek_output.taxa:
        inchi_keys = [resolve_name_to_inchi(c) for c in taxon.compounds]
        taxon.inchi_keys = [c for c in inchi_keys if c is not None]
        taxon.inchi_key_simps = list(set([simplify_inchi_key(c) for c in taxon.inchi_keys]))
    return deepseek_output


def add_all_extra_info_to_output(deepseek_output: TaxaData):
    add_accepted_info(deepseek_output)
    add_inchi_keys(deepseek_output)
    # print(deepseek_output)


if __name__ == '__main__':
    example = TaxaData(taxa=[Taxon(scientific_name='acanthochlamys bracteata p. c. kao',
                                   compounds=['tetracosanoic acid', 'stigmasterol', 'demethyl coniferin',
                                              'kaempferol 3-o-(3",6"-di-o-e-p-coumaroyl)-β-d-glcopyranoside',
                                              'palmitic acid', 'acanthochlamic acid',
                                              '28-feruloxyloctacosanoyl 1-glyceride', 'ayanin', 'ethyl caffeate',
                                              'stigmasta-5,22-dien-3β,7α-diol',
                                              'isorhamnetin 3-o-(6"-di-o-e-p-coumaroyl)-β-d-glucopyranoside',
                                              'euscaphic acid', 'heptacosan-1-ol', 'liquiritin',
                                              'isorhamnetin 3-o-(3",6"-di-o-e-p-coumaroyl)-β-d-glucopyranoside',
                                              'stigmasterol 3-o-β-d-glucopyranoside', 'isoliquiritigenin',
                                              'stigmasta-5,22-dien-3β,7β-diol', 'tiliroside', 'liquiritigenin'])]
                       )
    add_all_extra_info_to_output(example)
