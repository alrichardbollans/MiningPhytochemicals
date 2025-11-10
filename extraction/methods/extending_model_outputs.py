import os
import pickle
import time
import urllib

import cirpy
import pandas as pd
from phytochempy.compound_properties import simplify_inchi_key
from wcvpy.wcvp_download import get_all_taxa
from wcvpy.wcvp_name_matching import get_accepted_info_from_names_in_column, get_genus_from_full_name
from pubchempy import get_compounds

from data.get_wikidata import data_path, inchi_translation_cache
from extraction.methods.string_cleaning_methods import clean_compound_strings
from extraction.methods.structured_output_schema import TaxaData, Taxon

try:
    pkled_inchi_translation_result = pickle.load(open(inchi_translation_cache, 'rb'))
except FileNotFoundError:
    pkled_inchi_translation_result = {}

_original_timeout = 0.34
_timeout = [0.3]
# _wcvp_taxa = get_all_taxa()
def add_accepted_info(deepseek_output: TaxaData):
    deepseek_names = pd.DataFrame([c.scientific_name for c in deepseek_output.taxa], columns=['scientific_name'])
    acc_deepseek_names = get_accepted_info_from_names_in_column(deepseek_names, 'scientific_name', all_taxa=_wcvp_taxa)
    acc_deepseek_names = acc_deepseek_names.set_index('scientific_name')
    for taxon in deepseek_output.taxa:
        taxon.accepted_name = acc_deepseek_names.loc[taxon.scientific_name, 'accepted_name']
        taxon.accepted_species = acc_deepseek_names.loc[taxon.scientific_name, 'accepted_species']
        taxon.accepted_genus = get_genus_from_full_name(taxon.accepted_species)


def is_valid_inchikey(inchikey: str):
    """
    Check if an InChIKey has a valid format.
    """
    return inchikey and len(inchikey) == 27 and "-" in inchikey


def resolve_name_to_inchi(name: str):
    """

    """
    standard_name = clean_compound_strings(name)
    standard_name = standard_name.replace('β', 'beta')
    standard_name = standard_name.replace('α', 'alpha')
    standard_name = standard_name.replace('ψ', 'psi')
    standard_name = standard_name.replace('γ', 'gamma')
    standard_name = standard_name.replace('δ', 'delta')
    failed_search = False
    if standard_name not in pkled_inchi_translation_result:
        out = None
        if standard_name is not None and standard_name !='':

            try:
                time.sleep(_timeout[0])
                compounds = get_compounds(standard_name, 'name')
                if compounds is not None and len(compounds) > 0:
                    out = compounds[0].inchikey  # Take first result
                else:
                    # print(f"Name not found in PubChem: {standard_name}")

                    inch = cirpy.resolve(standard_name, 'stdinchikey')
                    if inch is not None:
                        out = inch.replace('InChIKey=', '')

                _timeout[0] = _original_timeout
            except (urllib.error.HTTPError, urllib.error.URLError):
                out = None
                failed_search = True
                print(f'WARNING: not resolved: {name}')
                _timeout[0] = _timeout[0]*2
        if out is not None:
            assert is_valid_inchikey(out)
        pkled_inchi_translation_result[standard_name] = out
    if not failed_search:
        with  open(inchi_translation_cache, 'wb') as pfile:
            pickle.dump(pkled_inchi_translation_result, pfile)
    return pkled_inchi_translation_result[standard_name]


def add_inchi_keys(deepseek_output: TaxaData):
    for taxon in deepseek_output.taxa:
        inchi_key_out_dict = {}
        inchi_key_simp_out_dict = {}
        for compound in taxon.compounds or []:
            inchi_key = resolve_name_to_inchi(compound)
            if inchi_key is not None:
                inchi_key_out_dict[compound] = inchi_key
                inchi_key_simp_out_dict[compound] = simplify_inchi_key(inchi_key)
        taxon.inchi_keys = inchi_key_out_dict
        taxon.inchi_key_simps = inchi_key_simp_out_dict

    return deepseek_output


def add_all_extra_info_to_output(deepseek_output: TaxaData):
    add_accepted_info(deepseek_output)
    add_inchi_keys(deepseek_output)
    # print(deepseek_output)


if __name__ == '__main__':
    pkled_result = pickle.load(open(inchi_translation_cache, 'rb'))
    print(pkled_result)
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
