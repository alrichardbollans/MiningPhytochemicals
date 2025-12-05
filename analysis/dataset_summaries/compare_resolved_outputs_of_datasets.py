import os
import pathlib

import pandas as pd

from analysis.dataset_summaries.data_summaries import get_deepseek_accepted_output_as_df, summarise
from data.get_colombian_data import get_sanitised_dois_for_colombian_papers
from data.get_data_with_full_texts import validation_data_csv
from data.get_knapsack_data import knapsack_plantae_compounds_csv
from data.get_papers_with_no_hits import get_sanitised_dois_for_papers
from data.get_wikidata import wikidata_plantae_compounds_csv, wikidata_plantae_reference_data_csv


def compare_two_outputs_accepted(df1, df2, out_tag: str, label1: str, label2: str):
    out_dir = os.path.join('comparisons', out_tag)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    df1 = df1.dropna(subset=['accepted_name', 'InChIKey_simp'], how='any')
    df2 = df2.dropna(subset=['accepted_name', 'InChIKey_simp'], how='any')
    df1['pairs'] = df1['accepted_name'] + df1['InChIKey_simp']
    df1 = df1.drop_duplicates(subset=['pairs'], keep='first')[['pairs']]
    df2['pairs'] = df2['accepted_name'] + df2['InChIKey_simp']
    df2 = df2.drop_duplicates(subset=['pairs'], keep='first')[['pairs']]

    merged = pd.merge(df1, df2, on=['pairs'], how='outer', indicator=True)
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

    result_venn_diagram(unique_left_count, unique_right_count, shared, os.path.join(out_dir, 'resolved_data_venn.jpg'),
                        label1, label2)
    return merged


def compare_two_outputs_verbatim(df1, df2, out_dir: str):
    raise NotImplementedError(
        'This doesnt match names properly. Need to use compare_method_outputs_on_specific_dois.py for this.')
    df1_verbatim = df1[['organism_name', 'example_compound_name']].drop_duplicates()
    df2_verbatim = df2[['organism_name', 'example_compound_name']].drop_duplicates()

    merged = pd.merge(df1_verbatim, df2_verbatim, on=['organism_name', 'example_compound_name'], how='outer',
                      indicator=True)
    print(merged)
    unique_left_count = len(merged[merged['_merge'] == 'left_only'])
    unique_right_count = len(merged[merged['_merge'] == 'right_only'])
    shared = len(merged[merged['_merge'] == 'both'])
    print(f'Unique in df1: {unique_left_count}')
    print(f'Unique in df2: {unique_right_count}')
    print(f'Shared: {shared}')
    print(f'Percentage shared: {shared / (unique_left_count + unique_right_count + shared) * 100:.2f}%')

    result_venn_diagram(unique_left_count, unique_right_count, shared, os.path.join(out_dir, 'verbatim_data_venn.jpg'))


def result_venn_diagram(found_in_1_but_not_2: int, found_in_2_but_not_1: int, overlap: int, outpath: str, label1: str,
                        label2: str):
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

    phytochem_txt_dir, result = get_sanitised_dois_for_papers('phytochemistry papers')
    deepseek_df = get_deepseek_accepted_output_as_df(result)
    compare_two_outputs_accepted(wikidata, deepseek_df, 'deepseek_on_phytochem_papers_vs_all_wikidata', 'WikiData',
                                 'DeepSeek')
    compare_two_outputs_accepted(knapsack_data, deepseek_df, 'deepseek_on_phytochem_papers_vs_all_knapsack', 'KNApSAcK',
                                 'DeepSeek')
    merged_data = compare_two_outputs_accepted(pd.concat([wikidata, knapsack_data]), deepseek_df,
                                               'deepseek_on_phytochem_papers_vs_all_wikidata_knapsack',
                                               'WikiData and KNApSAcK',
                                               'DeepSeek')
    only_in_deepseek_merge_info = merged_data[merged_data['_merge'] == 'right_only']
    deepseek_df['pairs'] = deepseek_df['accepted_name'] + deepseek_df['InChIKey_simp']
    only_in_deepseek = deepseek_df[
        deepseek_df['pairs'].isin(only_in_deepseek_merge_info['pairs'].values)]
    summarise(only_in_deepseek, 'deepseek_phytochem_papers_not_in_other_sources', output_data=True)


    ## Colombian data
    colombian_dois = list(get_sanitised_dois_for_colombian_papers().keys())
    colombian_data = get_deepseek_accepted_output_as_df(colombian_dois)
    species_to_collect = \
        pd.read_csv(os.path.join('..', 'data', 'colombian species not in datasets', 'species.csv'), index_col=0)[
            'accepted_species'].tolist()
    colombian_data = colombian_data[colombian_data['accepted_species'].isin(species_to_collect)]
    compare_two_outputs_accepted(pd.concat([wikidata, knapsack_data]), colombian_data,
                                 'deepseek_on_colombian_papers_vs_all_wikidata_knapsack',
                                 'WikiData and KNApSAcK',
                                 'DeepSeek')
    # With validation data
    doi_data_table = pd.read_csv(validation_data_csv, index_col=0)
    dois = doi_data_table['refDOI'].unique().tolist()
    wikidata = pd.read_csv(wikidata_plantae_reference_data_csv)
    deepseek_df = get_deepseek_accepted_output_as_df(dois)
    compare_two_outputs_accepted(wikidata[wikidata['refDOI'].isin(dois)], deepseek_df,
                                 'wikidata_deepseek_comparison_on_validation_data', 'WikiData',
                                 'DeepSeek')

    compare_two_outputs_accepted(wikidata, deepseek_df, 'deepseek_on_validation_data_vs_all_wikidata', 'WikiData',
                                 'DeepSeek')
    compare_two_outputs_accepted(knapsack_data, deepseek_df, 'deepseek_on_validation_data_vs_all_knapsack', 'KNApSAcK',
                                 'DeepSeek')


if __name__ == '__main__':
    main()
