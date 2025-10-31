import os

import pandas as pd

from data.get_data_with_full_texts import data_with_full_texts_csv, validation_data_csv, test_data_csv
from extraction.methods.get_agreements_and_disagreements import check_records_for_doi
from extraction.methods.structured_output_schema import TaxaData


def count_names_in_taxadata(taxadata: TaxaData, verbatim=True):
    names = []
    for taxon in taxadata.taxa:
        if verbatim:
            names.append(taxon.scientific_name)
        else:
            names.append(taxon.accepted_name)
    return len(set(names))


def count_pairs_in_taxadata(taxadata: TaxaData, verbatim=True):
    counts = 0
    for taxon in taxadata.taxa:
        if verbatim:
            counts += len(taxon.compounds)
        else:
            counts += len(taxon.inchi_key_simps)
    return counts


def get_counts_for_dois(dois: list, verbatim=True):
    wikidata_pair_counts = 0
    wikidata_name_counts = 0
    deepseek_pair_counts = 0
    deepseek_name_counts = 0
    agreements = 0
    found_in_wikidata_but_not_deepseek = 0
    found_in_deepseek_but_not_wikidata = 0
    for doi in dois:
        verbatim_results, accepted_results = check_records_for_doi(doi)
        if verbatim:
            results = verbatim_results
        else:
            results = accepted_results
        wikidata_pair_counts += count_pairs_in_taxadata(results[0], verbatim=verbatim)
        wikidata_name_counts += count_names_in_taxadata(results[0], verbatim=verbatim)
        deepseek_pair_counts += count_pairs_in_taxadata(results[1], verbatim=verbatim)
        deepseek_name_counts += count_names_in_taxadata(results[1], verbatim=verbatim)
        agreements += count_pairs_in_taxadata(results[2], verbatim=verbatim)
        found_in_wikidata_but_not_deepseek += count_pairs_in_taxadata(results[3], verbatim=verbatim)
        found_in_deepseek_but_not_wikidata += count_pairs_in_taxadata(results[4], verbatim=verbatim)

    total_pairs = found_in_wikidata_but_not_deepseek + found_in_deepseek_but_not_wikidata + agreements
    assert total_pairs == wikidata_pair_counts + deepseek_pair_counts - agreements
    print(
        f'{total_pairs} {wikidata_pair_counts} {deepseek_pair_counts} {agreements} {found_in_wikidata_but_not_deepseek} {found_in_deepseek_but_not_wikidata}')

    return {'total_pairs': total_pairs, 'wikidata_pair_counts': wikidata_pair_counts,
            'deepseek_pair_counts': deepseek_pair_counts, 'agreements': agreements,
            'found_in_wikidata_but_not_deepseek': found_in_wikidata_but_not_deepseek,
            'found_in_deepseek_but_not_wikidata': found_in_deepseek_but_not_wikidata,
            'wikidata_name_counts': wikidata_name_counts,
            'deepseek_name_counts': deepseek_name_counts}


def result_venn_diagram(result, outpath: str):
    # library
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2

    # Use the venn2 function
    venn2(subsets=(result['found_in_wikidata_but_not_deepseek'],
                   result['found_in_deepseek_but_not_wikidata'],
                   result['agreements']),
          set_labels=('WikiData Pairs', 'Deepseek Pairs'))
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    outpath = os.path.join('outputs', 'wikidata_deepseek_comparison')
    ## Validation Data
    doi_data_table = pd.read_csv(validation_data_csv, index_col=0)
    dois = doi_data_table['refDOI'].unique().tolist()
    results = get_counts_for_dois(dois)
    pd.DataFrame.from_dict(results,orient='index').to_csv(
        os.path.join(outpath, 'verbatim_validation_data_counts.csv'))
    result_venn_diagram(results, os.path.join(outpath, 'verbatim_validation_data_venn.jpg'))
    results = get_counts_for_dois(dois, verbatim=False)
    pd.DataFrame.from_dict(results,orient='index').to_csv(os.path.join(outpath, 'resolved_validation_data_counts.csv'))
    result_venn_diagram(results, os.path.join(outpath, 'resolved_validation_data_venn.jpg'))

    ## Test Data
    doi_data_table = pd.read_csv(test_data_csv, index_col=0)
    dois = doi_data_table['refDOI'].unique().tolist()
    results = get_counts_for_dois(dois)
    pd.DataFrame.from_dict(results, orient='index').to_csv(
        os.path.join(outpath, 'verbatim_test_data_counts.csv'))
    result_venn_diagram(results, os.path.join(outpath, 'verbatim_test_data_venn.jpg') )
    results = get_counts_for_dois(dois, verbatim=False)
    pd.DataFrame.from_dict(results, orient='index').to_csv(
        os.path.join(outpath, 'resolved_test_data_counts.csv'))
    result_venn_diagram(results, os.path.join(outpath, 'resolved_test_data_venn.jpg'))


if __name__ == '__main__':
    main()
