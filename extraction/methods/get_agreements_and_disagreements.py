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
    verbatim_matches = []
    unmatched_from_taxon1 = []
    unmatched_from_taxon2 = []
    for taxon1 in taxadata1.taxa:
        for taxon2 in taxadata2.taxa:
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
    for taxon1 in taxadata1.taxa:
        if not any(check_organism_names_match(taxon1.scientific_name, taxon2.scientific_name) for taxon2 in
                   taxadata2.taxa):
            unmatched_from_taxon1.append(taxon1)
    for taxon2 in taxadata2.taxa:
        if not any(check_organism_names_match(taxon2.scientific_name, taxon1.scientific_name) for taxon1 in
                   taxadata1.taxa):
            unmatched_from_taxon2.append(taxon2)

    return verbatim_matches, unmatched_from_taxon1, unmatched_from_taxon2


def get_accepted_matches(taxadata1: TaxaData, taxadata2: TaxaData):
    accepted_matches = []
    unmatched_from_taxon1 = []
    unmatched_from_taxon2 = []
    for taxon1 in taxadata1.taxa:
        for taxon2 in taxadata2.taxa:
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
    for taxon1 in taxadata1.taxa:
        if not any(check_organism_names_match(taxon1.accepted_name, taxon2.accepted_name) for taxon2 in
                   taxadata2.taxa):
            unmatched_from_taxon1.append(taxon1)
    for taxon2 in taxadata2.taxa:
        if not any(check_organism_names_match(taxon2.accepted_name, taxon1.accepted_name) for taxon1 in
                   taxadata1.taxa):
            unmatched_from_taxon2.append(taxon2)

    return accepted_matches, unmatched_from_taxon1, unmatched_from_taxon2


def convert_wikidata_table_to_taxadata(data_table: pd.DataFrame) -> TaxaData:
    taxa_output = []
    for organism in data_table['organism_name'].unique():
        organism_data = data_table[data_table['organism_name'] == organism]
        taxon = Taxon(scientific_name=organism, compounds=organism_data['example_compound_name'].unique().tolist(),
                      accepted_name=organism_data['accepted_name'].iloc[0],
                      inchikey_simps=organism_data['InChIKey_simp'].unique().tolist())
        taxa_output.append(taxon)
    return TaxaData(taxa=taxa_output)


def get_names_from_wikidata_that_match_deepseek(data_table, deepseek_output):
    data_table = data_table.drop_duplicates(subset=['organism_name'])
    verbatim_matches = defaultdict(dict)
    accepted_name_matches = defaultdict(dict)
    accepted_species_or_higher_matches = defaultdict(dict)
    for index, row in data_table.iterrows():
        organism_name = row['organism_name']
        verbatim_matches[organism_name]['taxa'] = []
        accepted_name_matches[organism_name]['taxa'] = []
        accepted_species_or_higher_matches[organism_name]['taxa'] = []
        for taxon in deepseek_output.taxa:
            if check_organism_names_match(organism_name, taxon.scientific_name):
                verbatim_matches[organism_name]['taxa'].append(taxon)
            if row['accepted_name'] == row['accepted_name'] and row['accepted_name'] != '' and row[
                'accepted_name'] == taxon.accepted_name:
                accepted_name_matches[organism_name]['taxa'].append(taxon)
            if row['accepted_species'] == row['accepted_species'] and row['accepted_name'] != '':
                if row['accepted_species'] == taxon.accepted_species:
                    accepted_species_or_higher_matches[organism_name]['taxa'].append(taxon)
            else:
                # WHere no accepted species given, check genus
                if row['accepted_genus'] == row['accepted_genus'] and row['accepted_genus'] != '':
                    if row['accepted_genus'] == taxon.accepted_genus:
                        accepted_species_or_higher_matches[organism_name]['taxa'].append(taxon)

    return verbatim_matches, accepted_name_matches, accepted_species_or_higher_matches


def get_compounds_from_wikidata_that_match_deepseek(organism_data_table: pd.DataFrame, deepseek_taxon_output: Taxon):
    assert len(organism_data_table['organism_name'].unique().tolist()) == 1
    organism_data_table = organism_data_table.drop_duplicates(subset=['example_compound_name', 'InChIKey_simp'])
    verbatim_matches = []
    inchikey_simp_matches = []
    for index, row in organism_data_table.iterrows():
        compound_name = row['example_compound_name']
        for deepseek_compound in deepseek_taxon_output.compounds:
            if check_compound_names_match(compound_name, deepseek_compound):
                verbatim_matches.append(deepseek_compound)

        for inchikeysimp in deepseek_taxon_output.inchi_key_simps:
            if row['InChIKey_simp'] == inchikeysimp:
                inchikey_simp_matches.append(inchikeysimp)

    return verbatim_matches, inchikey_simp_matches


def get_matches_between_wikidata_and_deepseek_for_doi(doi: str):
    # fulltext = get_txt_from_file(os.path.join(fulltext_dir, sanitise_doi(doi) + '.txt'))
    doi_data_table = pd.read_csv(data_with_full_texts_csv, index_col=0)

    doi_data_table = doi_data_table[doi_data_table['refDOI'] == doi]

    deepseek_output = pickle.load(open(os.path.join(deepseek_pkls_path, sanitise_doi(doi) + '.pkl'), 'rb'))

    verbatim_name_matches, accepted_name_matches, accepted_species_or_higher_matches = get_names_from_wikidata_that_match_deepseek(
        doi_data_table, deepseek_output)

    all_verbatim_name_match_details = defaultdict(list)
    for organism in verbatim_name_matches:
        organism_data = doi_data_table[doi_data_table['organism_name'] == organism]
        for relevant_deep_seek_taxon in verbatim_name_matches[organism]['taxa']:
            verbatim_compound_matches, inchikey_simp_matches = get_compounds_from_wikidata_that_match_deepseek(
                organism_data, relevant_deep_seek_taxon)
            match_details = Taxon(scientific_name=relevant_deep_seek_taxon.scientific_name,
                                  compounds=verbatim_compound_matches)
            match_details.inchi_key_simps = inchikey_simp_matches
            all_verbatim_name_match_details[organism].append(match_details)

    all_accepted_name_match_details = defaultdict(list)
    for organism in accepted_name_matches:
        organism_data = doi_data_table[doi_data_table['organism_name'] == organism]
        for relevant_deep_seek_taxon in accepted_name_matches[organism]['taxa']:
            verbatim_compound_matches, inchikey_simp_matches = get_compounds_from_wikidata_that_match_deepseek(
                organism_data, relevant_deep_seek_taxon)
            match_details = Taxon(scientific_name=relevant_deep_seek_taxon.scientific_name,
                                  compounds=verbatim_compound_matches)
            match_details.inchi_key_simps = inchikey_simp_matches
            all_accepted_name_match_details[organism].append(match_details)

    all_accepted_accepted_species_or_higher_match_details = defaultdict(list)
    for organism in accepted_species_or_higher_matches:
        organism_data = doi_data_table[doi_data_table['organism_name'] == organism]
        for relevant_deep_seek_taxon in accepted_species_or_higher_matches[organism]['taxa']:
            verbatim_compound_matches, inchikey_simp_matches = get_compounds_from_wikidata_that_match_deepseek(
                organism_data, relevant_deep_seek_taxon)
            match_details = Taxon(scientific_name=relevant_deep_seek_taxon.scientific_name,
                                  compounds=verbatim_compound_matches)
            match_details.inchi_key_simps = inchikey_simp_matches
            all_accepted_accepted_species_or_higher_match_details[organism].append(match_details)

    return all_verbatim_name_match_details, all_accepted_name_match_details, all_accepted_accepted_species_or_higher_match_details


def get_agreements_and_disagreements_for_doi(doi: str, match_details):
    deepseek_output = pickle.load(open(os.path.join(deepseek_pkls_path, sanitise_doi(doi) + '.pkl'), 'rb'))
    deepseek_cases_in_wikidata = []
    for taxon in deepseek_output.taxa:
        for organism in match_details:
            for matched_taxon in match_details[organism]:
                if taxon.scientific_name == matched_taxon.scientific_name:
                    deepseek_cases_in_wikidata.append(matched_taxon)

    unmatched_deepseek_names = []
    for taxon in deepseek_output.taxa:
        if not any(
                taxon.scientific_name == matched_taxon.scientific_name for matched_taxon in deepseek_cases_in_wikidata):
            unmatched_deepseek_names.append(taxon)
    unmatched_deepseek_name_compound_pairs = unmatched_deepseek_names[:]
    for taxon in deepseek_output.taxa:
        for matched_name_taxon in deepseek_cases_in_wikidata:
            if taxon.scientific_name == matched_name_taxon.scientific_name:
                relevant_taxon = Taxon(scientific_name=taxon.scientific_name, compounds=[])
                relevant_taxon.inchi_key_simps = []
                for compound in taxon.inchi_key_simps:
                    if compound not in matched_name_taxon.inchi_key_simps:
                        relevant_taxon.inchi_key_simps.append(compound)  #
                unmatched_deepseek_name_compound_pairs.append(relevant_taxon)

    doi_data_table = pd.read_csv(data_with_full_texts_csv, index_col=0)

    doi_data_table = doi_data_table[doi_data_table['refDOI'] == doi]
    unmatched_wikidata_name_compound_pairs = []
    for organism in doi_data_table['organism_name'].unique():
        organism_data = doi_data_table[doi_data_table['organism_name'] == organism]
        if organism not in match_details:
            relevant_taxon = Taxon(scientific_name=organism,
                                   compounds=organism_data['example_compound_name'].unique().tolist())
            relevant_taxon.inchi_key_simps = organism_data['InChIKey_simp'].unique().tolist()
            unmatched_wikidata_name_compound_pairs.append(relevant_taxon)
        else:
            matching_with_deepseek = match_details[organism]
            all_names_for_matching_compounds = []
            for t in matching_with_deepseek:
                all_names_for_matching_compounds.extend(t.compounds)
            inchi_keys_simps = []
            compounds = []
            for index, row in organism_data.iterrows():
                if not any(row['InChIKey_simp'] in matched_taxon.inchi_key_simps for matched_taxon in
                           matching_with_deepseek):
                    inchi_keys_simps.append(row['InChIKey_simp'])

                if not any(
                        check_compound_names_match(row['example_compound_name'], other_compound) for other_compound in
                        all_names_for_matching_compounds):
                    compounds.append(row['example_compound_name'])
            relevant_taxon = Taxon(scientific_name=organism,
                                   compounds=compounds)
            relevant_taxon.inchi_key_simps = inchi_keys_simps
            unmatched_wikidata_name_compound_pairs.append(relevant_taxon)

    return unmatched_wikidata_name_compound_pairs, unmatched_deepseek_name_compound_pairs


def deduplicate_taxa_list_on_accepted_name(taxa):
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
                    new_taxon.append(taxon.scientific_name)
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

    # work by matching on accepted names and inchikey simps
    doi_data_table = doi_data_table[['accepted_name', 'InChIKey_simp']].drop_duplicates()

    v, accepted_name_matches, s = get_matches_between_wikidata_and_deepseek_for_doi(doi)
    unmatched_wikidata_name_compound_pairs, unmatched_deepseek_name_compound_pairs = get_agreements_and_disagreements_for_doi(
        doi, accepted_name_matches)
    print(f'unmatched wikidata names: {len(unmatched_wikidata_name_compound_pairs)}')

    return doi_data_table, deepseek_output, accepted_name_matches, unmatched_wikidata_name_compound_pairs, unmatched_deepseek_name_compound_pairs


if __name__ == '__main__':
    check_records_for_doi('10.1002/CHIN.200549164')
