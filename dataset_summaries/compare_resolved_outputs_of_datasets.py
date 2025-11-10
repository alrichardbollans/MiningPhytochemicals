import os
import pathlib
import pickle

import pandas as pd

from data.get_data_with_full_texts import validation_data_csv
from data.get_knapsack_data import knapsack_plantae_compounds_csv
from data.get_papers_with_no_hits import get_sanitised_dois_for_papers
from data.get_wikidata import wikidata_plantae_compounds_csv
from data.parse_refs import sanitise_doi
from dataset_summaries.data_summaries import get_deepseek_accepted_output_as_df
from extraction.methods.get_agreements_and_disagreements import convert_taxadata_to_accepted_dataframe, convert_taxadata_to_verbatim_dataframe
from extraction.methods.running_models import deepseek_pkls_path


def compare_two_outputs_accepted(df1, df2, out_dir: str, label1: str, label2: str):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    df1_resolved = df1[['accepted_name', 'InChIKey_simp']].drop_duplicates()
    df2_resolved = df2[['accepted_name', 'InChIKey_simp']].drop_duplicates()

    merged = pd.merge(df1_resolved, df2_resolved, on=['accepted_name', 'InChIKey_simp'], how='outer', indicator=True)
    merged._merge.value_counts().reset_index().rename(
        columns={'index': '_merge', 0: 'count'}).to_csv(os.path.join(out_dir, 'resolved_data_summary.csv'))
    print(merged)
    unique_left_count = len(merged[merged['_merge'] == 'left_only'])
    unique_right_count = len(merged[merged['_merge'] == 'right_only'])
    shared = len(merged[merged['_merge'] == 'both'])
    print(f'Unique in df1: {unique_left_count}')
    print(f'Unique in df2: {unique_right_count}')
    print(f'Shared: {shared}')
    print(f'Percentage shared: {shared / (unique_left_count + unique_right_count + shared) * 100:.2f}%')

    result_venn_diagram(unique_left_count, unique_right_count, shared, os.path.join(out_dir, 'resolved_data_venn.jpg'), label1, label2)


def compare_two_outputs_verbatim(df1, df2, out_dir: str):
    raise NotImplementedError('This doesnt match names properly. Need to use compare_method_outputs_on_specific_dois.py for this.')
    df1_verbatim = df1[['organism_name', 'example_compound_name']].drop_duplicates()
    df2_verbatim = df2[['organism_name', 'example_compound_name']].drop_duplicates()

    merged = pd.merge(df1_verbatim, df2_verbatim, on=['organism_name', 'example_compound_name'], how='outer', indicator=True)
    print(merged)
    unique_left_count = len(merged[merged['_merge'] == 'left_only'])
    unique_right_count = len(merged[merged['_merge'] == 'right_only'])
    shared = len(merged[merged['_merge'] == 'both'])
    print(f'Unique in df1: {unique_left_count}')
    print(f'Unique in df2: {unique_right_count}')
    print(f'Shared: {shared}')
    print(f'Percentage shared: {shared / (unique_left_count + unique_right_count + shared) * 100:.2f}%')

    result_venn_diagram(unique_left_count, unique_right_count, shared, os.path.join(out_dir, 'verbatim_data_venn.jpg'))


def result_venn_diagram(found_in_1_but_not_2: int, found_in_2_but_not_1: int, overlap: int, outpath: str, label1: str, label2: str):
    # library
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2

    # Use the venn2 function
    venn2(subsets=(found_in_1_but_not_2,
                   found_in_2_but_not_1,
                   overlap),
          set_labels=(f'{label1} Pairs', f'{label2} Pairs'))
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    wikidata = pd.read_csv(wikidata_plantae_compounds_csv)
    knapsack_data = pd.read_csv(knapsack_plantae_compounds_csv)

    compare_two_outputs_accepted(wikidata, knapsack_data, 'wikidata_knapsack_comparison', 'WikiData', 'KNApSAcK')

    # With validation data
    doi_data_table = pd.read_csv(validation_data_csv, index_col=0)
    dois = doi_data_table['refDOI'].unique().tolist()
    deepseek_df = get_deepseek_accepted_output_as_df(dois)
    compare_two_outputs_accepted(wikidata[wikidata['refDOI'].isin(dois)], deepseek_df, 'wikidata_deepseek_comparison_on_validation_data', 'WikiData',
                                 'DeepSeek')

    compare_two_outputs_accepted(wikidata, deepseek_df, 'deepseek_on_validation_data_vs_all_wikidata', 'WikiData', 'DeepSeek')
    compare_two_outputs_accepted(knapsack_data, deepseek_df, 'deepseek_on_validation_data_vs_all_knapsack', 'KNApSAcK', 'DeepSeek')

    phytochem_txt_dir, result = get_sanitised_dois_for_papers('phytochemistry papers')
    deepseek_df = get_deepseek_accepted_output_as_df(result)
    compare_two_outputs_accepted(wikidata, deepseek_df, 'deepseek_on_phytochem_papers_vs_all_wikidata', 'WikiData', 'DeepSeek')
    compare_two_outputs_accepted(knapsack_data, deepseek_df, 'deepseek_on_phytochem_papers_vs_all_knapsack', 'KNApSAcK', 'DeepSeek')




if __name__ == '__main__':
    main()
