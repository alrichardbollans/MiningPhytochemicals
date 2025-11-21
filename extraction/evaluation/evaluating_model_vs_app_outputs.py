import os

import pandas as pd
from wcvpy.wcvp_name_matching import get_species_binomial_from_full_name, get_accepted_info_from_names_in_column

from data.get_wikidata import WCVP_VERSION
from extraction.methods.extending_model_outputs import resolve_name_to_inchi


def get_precision_scores(case):
    results = pd.read_csv(os.path.join('manual_matching_results', 'manual results', case, 'results.csv'))
    found_pairs = results[results['decision'] == 'Yes']
    not_found_pairs = results[results['decision'] == 'No']
    assert len(set(found_pairs['pkl_file'].tolist() + not_found_pairs['pkl_file'].tolist())) >= 9

    true_positives = len(found_pairs)
    false_positives = len(not_found_pairs)

    precision = true_positives / (true_positives + false_positives)

    print(f'Precision: {precision}')
    return precision


def main():

    deepseek_score = get_precision_scores('validation cases')
    out_df = pd.DataFrame({'model': ['deepseek'], 'precision': [deepseek_score],
                           'Notes': ['']})
    out_df.to_csv(os.path.join('outputs', 'model_scores_on_validation_data.csv'))

    deepseek_score = get_precision_scores('colombian papers')
    out_df = pd.DataFrame({'model': ['deepseek'], 'precision': [deepseek_score],
                           'Notes': ['']})
    out_df.to_csv(os.path.join('outputs', 'model_scores_on_colombian_papers.csv'))


if __name__ == '__main__':
    main()
