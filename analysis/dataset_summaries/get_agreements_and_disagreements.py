import json
import os
import pickle

import pandas as pd

from analysis.extraction.running_extraction import deepseek_jsons_path
from data.get_data_with_full_texts import data_with_full_texts_csv
from data.parse_refs import sanitise_doi
from phytochemMiner import TaxaData, check_organism_names_match, Taxon, check_compound_names_match


def get_verbatim_matches(wikidata_taxadata1: TaxaData, deepseek_taxadata2: TaxaData):
    wikidata_deduplicated_taxadata1 = deduplicate_taxa_list_on_scientific_name(wikidata_taxadata1)
    deepseek_deduplicated_taxadata2 = deduplicate_taxa_list_on_scientific_name(deepseek_taxadata2)
    verbatim_matches = []
    unmatched_from_taxon1 = []
    unmatched_from_taxon2 = []
    for taxon1 in wikidata_deduplicated_taxadata1.taxa:
        for taxon2 in deepseek_deduplicated_taxadata2.taxa:
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
    for taxon1 in wikidata_deduplicated_taxadata1.taxa:
        if not any(check_organism_names_match(taxon1.scientific_name, taxon2.scientific_name) for taxon2 in
                   deepseek_deduplicated_taxadata2.taxa):
            unmatched_from_taxon1.append(taxon1)
    for taxon2 in deepseek_deduplicated_taxadata2.taxa:
        if not any(check_organism_names_match(taxon2.scientific_name, taxon1.scientific_name) for taxon1 in
                   wikidata_deduplicated_taxadata1.taxa):
            unmatched_from_taxon2.append(taxon2)

    return wikidata_deduplicated_taxadata1, deepseek_deduplicated_taxadata2, TaxaData(taxa=verbatim_matches), TaxaData(
        taxa=unmatched_from_taxon1), TaxaData(taxa=unmatched_from_taxon2)


def get_accepted_matches(wiki_taxadata1: TaxaData, deepseek_taxadata2: TaxaData):
    wikidata_deduplicated_taxadata1 = deduplicate_taxa_list_on_accepted_name(wiki_taxadata1)
    deepseek_deduplicated_taxadata2 = deduplicate_taxa_list_on_accepted_name(deepseek_taxadata2)
    accepted_matches = []
    unmatched_from_taxadata1 = []
    unmatched_from_taxadata2 = []
    for taxon1 in wikidata_deduplicated_taxadata1.taxa:
        for taxon2 in deepseek_deduplicated_taxadata2.taxa:
            if check_organism_names_match(taxon1.accepted_name, taxon2.accepted_name):
                taxon_with_matches_between1and2 = Taxon(scientific_name=taxon1.scientific_name, compounds=[],
                                                        accepted_name=taxon1.accepted_name, inchi_key_simps={}, matched_names=taxon1.matched_names)
                unmatched_from_taxon1 = Taxon(scientific_name=taxon1.scientific_name, compounds=[],
                                              accepted_name=taxon1.accepted_name, inchi_key_simps={}, matched_names=taxon1.matched_names)
                unmatched_from_taxon2 = Taxon(scientific_name=taxon2.scientific_name, compounds=[],
                                              accepted_name=taxon2.accepted_name, inchi_key_simps={}, matched_names=taxon2.matched_names)

                for compound in taxon1.inchi_key_simps:
                    inchi_key_simp_1 = taxon1.inchi_key_simps[compound]
                    assert inchi_key_simp_1 is not None
                    if any(inchi_key_simp_1 == taxon2.inchi_key_simps[c2] for c2 in taxon2.inchi_key_simps):
                        taxon_with_matches_between1and2.inchi_key_simps[compound] = inchi_key_simp_1
                        taxon_with_matches_between1and2.compounds.append(compound)
                    else:
                        unmatched_from_taxon1.inchi_key_simps[compound] = inchi_key_simp_1
                        unmatched_from_taxon1.compounds.append(compound)
                for compound in taxon2.inchi_key_simps:
                    inchi_key_simp2 = taxon2.inchi_key_simps[compound]
                    if not any(inchi_key_simp2 == taxon1.inchi_key_simps[c1] for c1 in taxon1.inchi_key_simps):
                        unmatched_from_taxon2.inchi_key_simps[compound] = inchi_key_simp2
                        unmatched_from_taxon2.compounds.append(compound)
                accepted_matches.append(taxon_with_matches_between1and2)
                if len(unmatched_from_taxon1.inchi_key_simps) > 0:
                    unmatched_from_taxadata1.append(unmatched_from_taxon1)
                if len(unmatched_from_taxon2.inchi_key_simps) > 0:
                    unmatched_from_taxadata2.append(unmatched_from_taxon2)

    for taxon1 in wikidata_deduplicated_taxadata1.taxa:
        if not any(check_organism_names_match(taxon1.accepted_name, taxon2.accepted_name) for taxon2 in
                   deepseek_deduplicated_taxadata2.taxa):
            unmatched_from_taxadata1.append(taxon1)
    for taxon2 in deepseek_deduplicated_taxadata2.taxa:
        if not any(check_organism_names_match(taxon2.accepted_name, taxon1.accepted_name) for taxon1 in
                   wikidata_deduplicated_taxadata1.taxa):
            unmatched_from_taxadata2.append(taxon2)

    return wikidata_deduplicated_taxadata1, deepseek_deduplicated_taxadata2, TaxaData(taxa=accepted_matches), TaxaData(
        taxa=unmatched_from_taxadata1), TaxaData(taxa=unmatched_from_taxadata2)


def convert_wikidata_table_to_taxadata(data_table: pd.DataFrame) -> TaxaData:
    taxa_output = []
    for organism in data_table['organism_name'].unique():
        organism_data = data_table[data_table['organism_name'] == organism]
        taxon = Taxon(scientific_name=organism, compounds=organism_data['example_compound_name'].unique().tolist(),
                      accepted_name=organism_data['accepted_name'].iloc[0],
                      inchi_key_simps= pd.Series(organism_data.InChIKey_simp.values,index=organism_data.example_compound_name).to_dict())
        taxa_output.append(taxon)
    return TaxaData(taxa=taxa_output)


def convert_taxadata_to_accepted_dataframe(taxa_data) -> pd.DataFrame:
    out=[]
    for taxon in taxa_data.taxa:
        for i in taxon.inchi_key_simps:
            out.append([taxon.accepted_name, taxon.inchi_key_simps[i], taxon.scientific_name, i])

    return pd.DataFrame(out, columns=['accepted_name','InChIKey_simp', 'extracted_organism_name','extracted_compound_name'])

def convert_taxadata_to_verbatim_dataframe(taxa_data) -> pd.DataFrame:
    out=[]
    for taxon in taxa_data.taxa:
        for i in taxon.compounds:
            out.append([taxon.scientific_name, i])

    return pd.DataFrame(out, columns=['organism_name','example_compound_name'])


def deduplicate_taxa_list_on_scientific_name(taxadat: TaxaData):
    taxa = taxadat.taxa
    unique_scientific_names = []
    for taxon in taxa:
        if taxon.scientific_name is not None and taxon.scientific_name not in unique_scientific_names:
            unique_scientific_names.append(taxon.scientific_name)

    new_taxa_list = []
    for name in unique_scientific_names:
        new_taxon = Taxon(scientific_name=name, compounds=[])
        new_taxon.inchi_key_simps = {}
        new_taxon.accepted_name = None
        for taxon in taxa:
            if taxon.scientific_name == name:
                if new_taxon.accepted_name is None:
                    new_taxon.accepted_name = taxon.accepted_name
                for compound in taxon.inchi_key_simps:
                    if compound is not None and compound not in new_taxon.inchi_key_simps:
                        new_taxon.inchi_key_simps[compound] = taxon.inchi_key_simps[compound]
                for compound in taxon.compounds or []:
                    if compound not in new_taxon.compounds:
                        new_taxon.compounds.append(compound)

        new_taxa_list.append(new_taxon)
    return TaxaData(taxa=new_taxa_list)


def deduplicate_taxa_list_on_accepted_name(taxadat: TaxaData):
    taxa = taxadat.taxa
    unique_accepted_names = []
    for taxon in taxa:
        if taxon.accepted_name is not None and taxon.accepted_name not in unique_accepted_names:
            unique_accepted_names.append(taxon.accepted_name)

    new_taxa_list = []
    for name in unique_accepted_names:
        new_taxon = Taxon(scientific_name=None, compounds=[])
        new_taxon.accepted_name = name
        new_taxon.inchi_key_simps = {}
        new_taxon.matched_names = []
        for taxon in taxa:
            if taxon.accepted_name == name:
                if taxon.scientific_name not in new_taxon.matched_names:
                    new_taxon.matched_names.append(taxon.scientific_name)
                for compound in taxon.inchi_key_simps:
                    if compound is not None and compound not in new_taxon.inchi_key_simps:
                        new_taxon.inchi_key_simps[compound] = taxon.inchi_key_simps[compound]
                for compound in taxon.compounds:
                    if compound not in new_taxon.compounds:
                        new_taxon.compounds.append(compound)

        new_taxa_list.append(new_taxon)
    return TaxaData(taxa=new_taxa_list)


def check_records_for_doi(doi: str):
    json_dict = json.load(open(os.path.join(deepseek_jsons_path, sanitise_doi(doi) + '.json'), 'r'))
    deepseek_output = TaxaData.model_validate(json_dict)
    doi_data_table = pd.read_csv(data_with_full_texts_csv, index_col=0)
    doi_data_table = doi_data_table[doi_data_table['refDOI'] == doi]

    wikidata = convert_wikidata_table_to_taxadata(doi_data_table)

    verbatim_results = get_verbatim_matches(wikidata, deepseek_output)
    accepted_results = get_accepted_matches(wikidata, deepseek_output)

    return verbatim_results, accepted_results


if __name__ == '__main__':
    check_records_for_doi('10.1590/S0100-40422001000100006')
