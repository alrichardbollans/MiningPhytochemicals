import os

import pandas as pd
from phytochemMiner import resolve_name_to_smiles
from wcvpy.wcvp_name_matching import get_accepted_info_from_names_in_column

from data.get_wikidata import WCVP_VERSION
from data.parse_refs import desanitise_doi


def get_standardised_correct_results_for_lotus(result_csv_file, function_to_convert_json_filename_to_DOI: callable):
    manually_checked_results = pd.read_csv(
        result_csv_file)
    manually_checked_results = manually_checked_results[manually_checked_results['decision'] == 'Yes']
    manually_checked_results = manually_checked_results.rename(columns={'taxon_name': 'organism_name'})

    if 'accepted_name' not in manually_checked_results.columns:
        for_lotus = get_accepted_info_from_names_in_column(manually_checked_results, 'organism_name',
                                                           wcvp_version=WCVP_VERSION)
    else:
        for_lotus = manually_checked_results

    for_lotus['chemical_entity_smiles'] = for_lotus['compound_name'].apply(resolve_name_to_smiles)
    for_lotus['reference_doi'] = for_lotus['json_file'].apply(function_to_convert_json_filename_to_DOI)

    # ## To upload to LOTUS

    for_lotus = for_lotus[['accepted_name', 'compound_name', 'chemical_entity_smiles', 'reference_doi']]

    # rename according to https://github.com/lotusnprod/lotus-o3?tab=readme-ov-file#usage
    for_lotus = for_lotus.rename(columns={'compound_name': 'chemical_entity_name',
                                          'accepted_name': 'taxon_name'})

    for_lotus = for_lotus.dropna(how='any')

    return for_lotus


def main():
    def convert_json_filename_to_DOI(json_filename):
        return desanitise_doi(json_filename.strip('_not_in_WD_KN.json'))

    standardised_results = get_standardised_correct_results_for_lotus(
        os.path.join('..','evaluate_deepseek_performance','manual_matching_results','manual results after accepted filter', 'pchem hits not in WD or KN', 'tmpb_78d0y5.csv'), convert_json_filename_to_DOI)
    print(standardised_results)
    standardised_results.to_csv('submissions_to_lotus/pchem_hits_for_lotus1.csv')


if __name__ == '__main__':
    main()
