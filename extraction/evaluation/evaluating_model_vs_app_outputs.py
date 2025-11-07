import os

import pandas as pd
from wcvpy.wcvp_name_matching import get_species_binomial_from_full_name, get_accepted_info_from_names_in_column

from data.get_wikidata import WCVP_VERSION
from extraction.methods.extending_model_outputs import resolve_name_to_inchi


def get_what_would_be_output_data():
    found_pairs = pd.read_csv(os.path.join('..', 'app_to_manually_check_methods', 'outputs', 'deepseek', 'found_pairs.csv'))
    found_pairs['InChIKey'] = found_pairs['compound'].apply(resolve_name_to_inchi)
    acc_data = get_accepted_info_from_names_in_column(found_pairs,'name', wcvp_version=WCVP_VERSION)
    acc_data.to_csv(os.path.join('outputs','manually checked output data', 'dataset.csv'), index=False)
    acc_data.describe(include='all').to_csv(os.path.join('outputs','manually checked output data', 'dataset_summary.csv'))

    resolved_dataset= acc_data.dropna(subset=['accepted_name', 'InChIKey'])
    resolved_dataset.to_csv(os.path.join('outputs','manually checked output data', 'dataset_resolved.csv'), index=False)
    resolved_dataset.describe(include='all').to_csv(os.path.join('outputs','manually checked output data', 'dataset_resolved_summary.csv'))


def get_precision_scores(model):
    found_pairs = pd.read_csv(os.path.join('..', 'app_to_manually_check_methods','outputs', model, 'found_pairs.csv'))
    not_found_pairs = pd.read_csv(os.path.join('..', 'app_to_manually_check_methods','outputs', model, 'not_found_pairs.csv'))
    assert len(set(found_pairs['doi'].tolist() + not_found_pairs['doi'].tolist())) >= 9

    true_positives = len(found_pairs)
    false_positives = len(not_found_pairs)

    precision = true_positives / (true_positives + false_positives)

    print(f'Precision: {precision}')
    return precision


def pseudorecall():
    found_pairs = pd.read_csv(os.path.join('..', 'app_to_manually_check_methods','outputs', 'deepseek', 'found_pairs.csv'))
    wikidata_found_pairs = pd.read_csv(os.path.join('..', 'app_to_manually_check_methods','outputs', 'wikidata', 'found_pairs.csv'))

    all_pairs = pd.concat([found_pairs, wikidata_found_pairs])
    all_pairs['name'] = all_pairs['name'].str.lower()
    all_pairs['name'] = all_pairs['name'].apply(get_species_binomial_from_full_name)
    all_pairs['compound'] = all_pairs['compound'].str.lower()
    all_pairs = all_pairs.drop_duplicates(subset=['name', 'compound', 'doi'])

    deepseek_recall = len(found_pairs) / len(all_pairs)
    wikidata_recall = len(wikidata_found_pairs) / len(all_pairs)

    print(f'deepseek pseudorecall: {deepseek_recall}')
    return deepseek_recall, wikidata_recall


def main():
    get_what_would_be_output_data()
    deepseek_recall, wikidata_recall = pseudorecall()

    deepseek_score = get_precision_scores('deepseek')
    # Note for wikidata, this is marked for verbatim matching which is not a fair comparison
    wikidata_score = get_precision_scores('wikidata')
    out_df = pd.DataFrame({'model': ['deepseek', 'wikidata'], 'precision': [deepseek_score, wikidata_score],
                           'Pseudorecall': [deepseek_recall, wikidata_recall],
                           'Notes': ['',
                                     'for wikidata, this is marked for verbatim matching which is not a fair comparison']})
    out_df.to_csv(os.path.join( 'outputs', 'model_scores_on_validation_data.csv'))


if __name__ == '__main__':
    main()
