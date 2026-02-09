import json
import os

import pandas as pd
from phytochemMiner import TaxaData

from analysis.summaries_and_comparisons_of_datasets_and_extractions.get_agreements_and_disagreements import get_verbatim_matches
from analysis.extraction_outputs.running_extraction import deepseek_jsons_path
from data.get_papers_with_no_hits import get_sanitised_dois_for_papers
from data.get_wikidata import data_path
from data.parse_refs import sanitise_doi


def get_errors_from_result(result):
    pair_errors = []
    taxon_errors = []
    taxa_without_compounds_errors = []
    for sanitised_doi in result:
        json_dict = json.load(open(os.path.join(deepseek_jsons_path, sanitised_doi + '.json'), 'r'))
        deepseek_output = TaxaData.model_validate(json_dict)
        verbatim_cases = get_verbatim_matches(deepseek_output, deepseek_output)
        deduplicated_output = verbatim_cases[0]
        for taxon in deduplicated_output.taxa:
            for compound in taxon.compounds:
                pair_errors.append([taxon.scientific_name, compound, sanitised_doi])
            taxon_errors.append([taxon.scientific_name, sanitised_doi])
            if len(taxon.compounds) == 0:
                taxa_without_compounds_errors.append([taxon.scientific_name, sanitised_doi])
    print(f'pair_errors:{pair_errors}')
    print(taxon_errors)
    print(taxa_without_compounds_errors)

    out_df = pd.DataFrame({'model': ['deepseek'], 'number_of_papers': [len(result)],
                           'Number of erroneous pairs': [len(pair_errors)],
                           'Number of taxa (which may be in text)': [len(taxon_errors)],
                           'Number of taxa extracted without associated compounds (which may be in text)': [
                               len(taxa_without_compounds_errors)]
                           })

    return out_df


def random_cases():
    random_txt_dir, result = get_sanitised_dois_for_papers('random papers')
    out_df = get_errors_from_result(result)
    out_df.to_csv(os.path.join('outputs', 'model_errors_on_random_data.csv'))


def medplant_cases():
    random_txt_dir, result = get_sanitised_dois_for_papers('medplant papers')
    out_df = get_errors_from_result(result)
    out_df.to_csv(os.path.join('outputs', 'model_errors_on_medplant_data.csv'))

    med_plant_given_data = pd.read_csv(os.path.join(data_path, 'medicinals_top_10000.csv')).dropna(subset=['DOI'])
    med_plant_given_data['sanitised_dois'] = med_plant_given_data['DOI'].apply(sanitise_doi)
    grouped = med_plant_given_data.groupby('sanitised_dois')['DOI'].nunique()
    assert grouped.max() == 1

    relevant_data = med_plant_given_data[med_plant_given_data['sanitised_dois'].isin(result)]
    relevant_data = relevant_data.drop_duplicates(subset=['DOI'])
    relevant_data = relevant_data[
        ['DOI', 'title', 'sanitised_dois', 'plant_species_binomials_unique_total', 'plant_species_binomials_counts']]
    # print(relevant_data)
    assert len(relevant_data[relevant_data['plant_species_binomials_unique_total'] > 0]) == len(relevant_data)
    relevant_data.to_csv(os.path.join('outputs', 'medplant_data_info.csv'))


def main():
    random_cases()
    medplant_cases()


if __name__ == '__main__':
    main()
