import os

import pandas as pd


def get_precision_scores(case):
    results = pd.read_csv(os.path.join('manual_matching_results', 'manual results', case, 'results.csv'))

    if case == 'colombian papers':
        species_to_collect = \
            pd.read_csv(os.path.join('..', '..', 'data', 'colombian species not in datasets', 'species.csv'), index_col=0)[
                'accepted_species'].tolist()
        results['colombian_species'] = results['taxon_name'].apply(
            lambda x: True if any(sp.lower() in x.lower() for sp in species_to_collect) else False)
        results = results[results['colombian_species']]



    found_pairs = results[results['decision'] == 'Yes']
    not_found_pairs = results[results['decision'] == 'No']
    problem_compounds = not_found_pairs['compound_name'].unique().tolist()
    assert len(set(found_pairs['json_file'].tolist() + not_found_pairs['json_file'].tolist())) >= 7

    true_positives = len(found_pairs)
    false_positives = len(not_found_pairs)

    precision = true_positives / (true_positives + false_positives)

    print(f'Precision: {precision}')
    return precision, true_positives, false_positives, true_positives + false_positives,problem_compounds


def main():
    deepseek_score, true_positives, false_positives, total,problem_compounds = get_precision_scores('validation cases')
    out_df = pd.DataFrame({'model': ['deepseek'], 'precision': [deepseek_score], 'total extracted pairs': [total],
                           'true_positives': [true_positives],
                           'false_positives': [false_positives],
                           'problem_compounds': [problem_compounds],
                           'Notes': ['']})
    out_df.to_csv(os.path.join('outputs', 'model_scores_on_validation_data.csv'))

    deepseek_score, true_positives, false_positives, total,problem_compounds = get_precision_scores('colombian papers')
    out_df = pd.DataFrame({'model': ['deepseek'], 'precision': [deepseek_score], 'total extracted pairs': [total]
                              , 'true_positives': [true_positives],
                           'false_positives': [false_positives],  'problem_compounds': [problem_compounds],'Notes': ['']})
    out_df.to_csv(os.path.join('outputs', 'model_scores_on_colombian_papers.csv'))


if __name__ == '__main__':
    main()
