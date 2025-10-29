import os
import pickle
from collections import defaultdict

import pandas as pd

from data.get_data_with_full_texts import data_with_full_texts_csv
from data.parse_refs import sanitise_doi
from extraction.methods.running_models import deepseek_pkls_path
from extraction.methods.string_matching_methods import check_organism_names_match, check_compound_names_match
from extraction.methods.structured_output_schema import Taxon, TaxaData


def get_verbatim_matches(taxadata1: TaxaData, taxadata2: TaxaData):
    deduplicated_taxadata1 = deduplicate_taxa_list_on_scientific_name(taxadata1)
    deduplicated_taxadata2 = deduplicate_taxa_list_on_scientific_name(taxadata2)
    verbatim_matches = []
    unmatched_from_taxon1 = []
    unmatched_from_taxon2 = []
    for taxon1 in deduplicated_taxadata1.taxa:
        for taxon2 in deduplicated_taxadata2.taxa:
            if check_organism_names_match(taxon1.scientific_name, taxon2.scientific_name):
                relevant_taxon = Taxon(scientific_name=taxon1.scientific_name, compounds=[])
                unmatched_taxon1 = Taxon(scientific_name=taxon1.scientific_name, compounds=[])
                unmatched_taxon2 = Taxon(scientific_name=taxon2.scientific_name, compounds=[])
                for compound in taxon1.compounds:
                    if any(check_compound_names_match(compound, compound2) for compound2 in taxon2.compounds):
                        relevant_taxon.compounds.append(compound)
                    else:
                        unmatched_taxon1.compounds.append(compound)
                for compound in taxon2.compounds:
                    if not any(check_compound_names_match(compound, compound2) for compound2 in taxon1.compounds):
                        unmatched_taxon2.compounds.append(compound)
                verbatim_matches.append(relevant_taxon)
                if len(unmatched_taxon1.compounds) > 0:
                    unmatched_from_taxon1.append(unmatched_taxon1)
                if len(unmatched_taxon2.compounds) > 0:
                    unmatched_from_taxon2.append(unmatched_taxon2)
    for taxon1 in deduplicated_taxadata1.taxa:
        if not any(check_organism_names_match(taxon1.scientific_name, taxon2.scientific_name) for taxon2 in
                   deduplicated_taxadata2.taxa):
            unmatched_from_taxon1.append(taxon1)
    for taxon2 in deduplicated_taxadata2.taxa:
        if not any(check_organism_names_match(taxon2.scientific_name, taxon1.scientific_name) for taxon1 in
                   deduplicated_taxadata1.taxa):
            unmatched_from_taxon2.append(taxon2)

    return deduplicated_taxadata1, deduplicated_taxadata2, TaxaData(taxa=verbatim_matches), TaxaData(
        taxa=unmatched_from_taxon1), TaxaData(taxa=unmatched_from_taxon2)


def get_accepted_matches(taxadata1: TaxaData, taxadata2: TaxaData):
    deduplicated_taxadata1 = deduplicate_taxa_list_on_accepted_name(taxadata1)
    deduplicated_taxadata2 = deduplicate_taxa_list_on_accepted_name(taxadata2)
    accepted_matches = []
    unmatched_from_taxon1 = []
    unmatched_from_taxon2 = []
    for taxon1 in deduplicated_taxadata1.taxa:
        for taxon2 in deduplicated_taxadata2.taxa:
            if check_organism_names_match(taxon1.accepted_name, taxon2.accepted_name):
                relevant_taxon = Taxon(scientific_name=taxon1.scientific_name, compounds=[],
                                       accepted_name=taxon1.accepted_name, inchi_key_simps=[])
                unmatched_taxon1 = Taxon(scientific_name=taxon1.scientific_name, compounds=[],
                                         accepted_name=taxon1.accepted_name, inchi_key_simps=[])
                unmatched_taxon2 = Taxon(scientific_name=taxon2.scientific_name, compounds=[],
                                         accepted_name=taxon2.accepted_name, inchi_key_simps=[])
                for compound in taxon1.inchi_key_simps:
                    if any(compound == compound2 for compound2 in taxon2.inchi_key_simps):
                        relevant_taxon.inchi_key_simps.append(compound)
                    else:
                        unmatched_taxon1.inchi_key_simps.append(compound)
                for compound in taxon2.inchi_key_simps:
                    if not any(compound == compound2 for compound2 in taxon1.inchi_key_simps):
                        unmatched_taxon2.inchi_key_simps.append(compound)
                accepted_matches.append(relevant_taxon)
                if len(unmatched_taxon1.inchi_key_simps) > 0:
                    unmatched_from_taxon1.append(unmatched_taxon1)
                if len(unmatched_taxon2.inchi_key_simps) > 0:
                    unmatched_from_taxon2.append(unmatched_taxon2)
    for taxon1 in deduplicated_taxadata1.taxa:
        if not any(check_organism_names_match(taxon1.accepted_name, taxon2.accepted_name) for taxon2 in
                   deduplicated_taxadata2.taxa):
            unmatched_from_taxon1.append(taxon1)
    for taxon2 in deduplicated_taxadata2.taxa:
        if not any(check_organism_names_match(taxon2.accepted_name, taxon1.accepted_name) for taxon1 in
                   deduplicated_taxadata1.taxa):
            unmatched_from_taxon2.append(taxon2)

    return deduplicated_taxadata1, deduplicated_taxadata2, TaxaData(taxa=accepted_matches), TaxaData(
        taxa=unmatched_from_taxon1), TaxaData(taxa=unmatched_from_taxon2)


def convert_wikidata_table_to_taxadata(data_table: pd.DataFrame) -> TaxaData:
    taxa_output = []
    for organism in data_table['organism_name'].unique():
        organism_data = data_table[data_table['organism_name'] == organism]
        taxon = Taxon(scientific_name=organism, compounds=organism_data['example_compound_name'].unique().tolist(),
                      accepted_name=organism_data['accepted_name'].iloc[0],
                      inchi_key_simps=organism_data['InChIKey_simp'].unique().tolist())
        taxa_output.append(taxon)
    return TaxaData(taxa=taxa_output)


def deduplicate_taxa_list_on_scientific_name(taxadat: TaxaData):
    taxa = taxadat.taxa
    unique_scientific_names = []
    for taxon in taxa:
        if taxon.scientific_name is not None and taxon.scientific_name not in unique_scientific_names:
            unique_scientific_names.append(taxon.scientific_name)

    new_taxa_list = []
    for name in unique_scientific_names:
        new_taxon = Taxon(scientific_name=name, compounds=[])
        new_taxon.inchi_key_simps = []
        new_taxon.accepted_name = None
        for taxon in taxa:
            if taxon.scientific_name == name:
                if new_taxon.accepted_name is None:
                    new_taxon.accepted_name = taxon.accepted_name
                for compound in taxon.inchi_key_simps:
                    if compound not in new_taxon.inchi_key_simps:
                        new_taxon.inchi_key_simps.append(compound)
                for compound in taxon.compounds:
                    if compound not in new_taxon.compounds:
                        new_taxon.compounds.append(compound)

        new_taxa_list.append(new_taxon)
    return TaxaData(taxa=new_taxa_list)


def deduplicate_taxa_list_on_accepted_name(taxadat: TaxaData):
    taxa = taxadat.taxa
    unique_scientific_names = []
    for taxon in taxa:
        if taxon.accepted_name is not None and taxon.accepted_name not in unique_scientific_names:
            unique_scientific_names.append(taxon.accepted_name)

    new_taxa_list = []
    for name in unique_scientific_names:
        new_taxon = Taxon(scientific_name=None, compounds=[])
        new_taxon.accepted_name = name
        new_taxon.inchi_key_simps = []
        new_taxon.matched_names = []
        for taxon in taxa:
            if taxon.accepted_name == name:
                if taxon.scientific_name not in new_taxon.matched_names:
                    new_taxon.matched_names.append(taxon.scientific_name)
                for compound in taxon.inchi_key_simps:
                    if compound not in new_taxon.inchi_key_simps:
                        new_taxon.inchi_key_simps.append(compound)
                for compound in taxon.compounds:
                    if compound not in new_taxon.compounds:
                        new_taxon.compounds.append(compound)

        new_taxa_list.append(new_taxon)
    return TaxaData(taxa=new_taxa_list)


def check_records_for_doi(doi: str):
    deepseek_output = pickle.load(open(os.path.join(deepseek_pkls_path, sanitise_doi(doi) + '.pkl'), 'rb'))
    doi_data_table = pd.read_csv(data_with_full_texts_csv, index_col=0)
    doi_data_table = doi_data_table[doi_data_table['refDOI'] == doi]

    wikidata = convert_wikidata_table_to_taxadata(doi_data_table)

    verbatim_results = get_verbatim_matches(wikidata, deepseek_output)
    accepted_results = get_accepted_matches(wikidata, deepseek_output)

    return verbatim_results, accepted_results


if __name__ == '__main__':
    check_records_for_doi('10.1002/CHIN.200549164')
