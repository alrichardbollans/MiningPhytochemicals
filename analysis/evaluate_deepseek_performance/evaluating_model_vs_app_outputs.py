import os

import pandas as pd


def get_precision_scores(case, with_filter: bool):
    if with_filter:
        results = pd.read_csv(os.path.join('manual_matching_results', 'manual results after accepted filter', case, 'results.csv'))
    else:
        results = pd.read_csv(os.path.join('manual_matching_results', 'manual results without filter', case, 'results.csv'))

    # if case == 'colombian papers':
    #     species_to_collect = \
    #         pd.read_csv(os.path.join('..', '..', 'data', 'colombian species not in datasets', 'species.csv'), index_col=0)[
    #             'accepted_species'].tolist()
    #     results['colombian_species'] = results['taxon_name'].apply(
    #         lambda x: True if any(sp.lower() in x.lower() for sp in species_to_collect) else False)
    #     results = results[results['colombian_species']]

    found_pairs = results[results['decision'] == 'Yes']
    not_found_pairs = results[results['decision'] != 'Yes']

    problem_compounds = not_found_pairs['compound_name'].unique().tolist()
    assert len(set(found_pairs['json_file'].tolist() + not_found_pairs['json_file'].tolist())) >= 7

    true_positives = len(found_pairs)
    false_positives = len(not_found_pairs)
    total = true_positives + false_positives
    assert total == len(results)
    precision = true_positives / total

    print(f'Precision: {precision}')

    ref_count = len(results['json_file'].unique().tolist())

    out_df = pd.DataFrame({'model': ['deepseek'], 'precision': [precision], 'total extracted pairs': [total]
                              , 'true_positives': [true_positives],
                           'false_positives': [false_positives], 'problem_compounds': [problem_compounds],
                           'ref_count': [ref_count], 'Notes': ['']})

    return out_df


def analyse_cases(with_filter: bool):
    out_df = get_precision_scores(
        'pchem hits not in WD or KN', with_filter=True)

    out_df.to_csv(os.path.join('outputs after accepted filter', 'model_scores_on_phytochem_papers_not_in_other_sources.csv'))

    if with_filter:
        out_dir = os.path.join('outputs after accepted filter')
    else:
        out_dir = os.path.join('outputs without filter')

    out_df = get_precision_scores(
        'validation cases', with_filter)

    out_df.to_csv(os.path.join(out_dir, 'model_scores_on_validation_data.csv'))

    out_df = get_precision_scores(
        'colombian papers', with_filter)
    out_df.to_csv(os.path.join(out_dir, 'model_scores_on_colombian_papers.csv'))


def main():
    analyse_cases(with_filter=True)
    analyse_cases(with_filter=False)


if __name__ == '__main__':
    main()
