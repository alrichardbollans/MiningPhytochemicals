import os
import pickle

import pandas as pd

from data.get_compound_occurences import data_path
from data.get_papers_with_no_hits import get_sanitised_dois_for_random_papers, get_sanitised_dois_for_medplant_papers
from data.parse_refs import sanitise_doi
from extraction.methods.get_agreements_and_disagreements import get_verbatim_matches
from extraction.methods.running_models import deepseek_pkls_path


def random_cases():
    random_txt_dir, result = get_sanitised_dois_for_random_papers()
    errors = []
    for sanitised_doi in result:
        deepseek_output = pickle.load(open(os.path.join(deepseek_pkls_path, sanitised_doi + '.pkl'), 'rb'))
        verbatim_cases = get_verbatim_matches(deepseek_output,deepseek_output)
        deduplicated_output = verbatim_cases[0]
        for taxon in deduplicated_output.taxa:
            for compound in taxon.compounds:
                errors.append((taxon.scientific_name, compound, sanitised_doi))
    print(errors)
    out_df = pd.DataFrame({'model': ['deepseek'], 'number_of_papers': [len(result)],
                           'Number of errors': [len(errors)],
                          })
    print(out_df)
    out_df.to_csv(os.path.join('outputs', 'model_errors_on_random_data.csv'))


def medplant_cases():
    random_txt_dir, result = get_sanitised_dois_for_medplant_papers()
    errors = []
    for sanitised_doi in result:
        deepseek_output = pickle.load(open(os.path.join(deepseek_pkls_path, sanitised_doi + '.pkl'), 'rb'))
        verbatim_cases = get_verbatim_matches(deepseek_output,deepseek_output)
        deduplicated_output = verbatim_cases[0]
        for taxon in deduplicated_output.taxa:
            for compound in taxon.compounds:
                errors.append([taxon.scientific_name, compound, sanitised_doi])
    print(errors)

    med_plant_given_data = pd.read_csv(os.path.join(data_path, 'medicinals_top_10000.csv')).dropna(subset=['DOI'])
    med_plant_given_data['sanitised_dois'] = med_plant_given_data['DOI'].apply(sanitise_doi)
    grouped = med_plant_given_data.groupby('sanitised_dois')['DOI'].nunique()
    assert grouped.max() == 1

    relevant_data = med_plant_given_data[med_plant_given_data['sanitised_dois'].isin(result)]
    relevant_data = relevant_data.drop_duplicates(subset=['DOI'])
    relevant_data = relevant_data[['DOI','title', 'sanitised_dois', 'plant_species_binomials_unique_total', 'plant_species_binomials_counts']]
    print(relevant_data)
    assert len(relevant_data[relevant_data['plant_species_binomials_unique_total']>0]) == len(relevant_data)
    relevant_data.to_csv(os.path.join('outputs', 'medplant_data_info.csv'))


    out_df = pd.DataFrame({'model': ['deepseek'], 'number_of_papers': [len(result)],
                           'Number of errors': [len(errors)], 'Errors': [str(errors)],
                           })
    out_df.to_csv(os.path.join('outputs', 'model_errors_on_medplant_data.csv'))

def main():
    # random_cases()
    medplant_cases()

if __name__ == '__main__':
    main()