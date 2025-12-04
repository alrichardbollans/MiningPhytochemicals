import pandas as pd
from phytochempy.compound_properties import simplify_inchi_key
from wcvpy.wcvp_name_matching import get_accepted_info_from_names_in_column

from data.get_wikidata import WCVP_VERSION
from extraction.methods.extending_model_outputs import resolve_name_to_inchi, resolve_name_to_smiles


def get_standardised_correct_results(result_csv_file):
    manually_checked_results = pd.read_csv(
        result_csv_file)
    manually_checked_results = manually_checked_results[manually_checked_results['decision'] == 'Yes']
    manually_checked_results = manually_checked_results.rename(columns={'taxon_name': 'organism_name'})
    acc_deepseek_df = get_accepted_info_from_names_in_column(manually_checked_results, 'organism_name', wcvp_version=WCVP_VERSION)
    acc_deepseek_df['InChIKey'] = acc_deepseek_df['compound_name'].apply(resolve_name_to_inchi)
    acc_deepseek_df['InChIKey_simp'] = acc_deepseek_df['InChIKey'].apply(simplify_inchi_key)
    acc_deepseek_df['SMILES'] = acc_deepseek_df['compound_name'].apply(resolve_name_to_smiles)
    acc_deepseek_df['DOI'] = acc_deepseek_df['pkl_file'].apply(lambda x: x.replace('_', '/').strip('.pkl'))
    return acc_deepseek_df

if __name__ == '__main__':
    name = 'hi.csv'
    if not (name.endswith(".csv") or name.endswith(".tsv")):
        print("Not a csv or tsv file")