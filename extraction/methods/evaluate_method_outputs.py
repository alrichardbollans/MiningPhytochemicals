import pandas as pd

from data.get_data_with_full_texts import data_with_full_texts_csv
from extraction.methods.get_agreements_and_disagreements import check_records_for_doi
from extraction.methods.structured_output_schema import TaxaData


def count_pairs_in_taxadata(taxadata:TaxaData, verbatim=True):
    counts = 0
    for taxon in taxadata.taxa:
        if verbatim:
            counts += len(taxon.compounds)
        else:
            counts += len(taxon.inchi_key_simps)
    return counts

def get_counts_for_dois(dois: list, verbatim=True):
    wikidata_pair_counts = 0
    deepseek_pair_counts = 0
    agreements = 0
    found_in_wikidata_but_not_deepseek = 0
    found_in_deepseek_but_not_wikidata = 0
    for doi in dois:
        try:
            verbatim_results, accepted_results = check_records_for_doi(doi)
            if verbatim:
                results = verbatim_results
            else:
                results = accepted_results
            wikidata_pair_counts += count_pairs_in_taxadata(results[0], verbatim=verbatim)
            deepseek_pair_counts += count_pairs_in_taxadata(results[1], verbatim=verbatim)
            agreements += count_pairs_in_taxadata(results[2], verbatim=verbatim)
            found_in_wikidata_but_not_deepseek += count_pairs_in_taxadata(results[3], verbatim=verbatim)
            found_in_deepseek_but_not_wikidata += count_pairs_in_taxadata(results[4], verbatim=verbatim)
        except FileNotFoundError as e:
            pass

    total_pairs = found_in_wikidata_but_not_deepseek + found_in_deepseek_but_not_wikidata + agreements
    assert total_pairs ==  wikidata_pair_counts + deepseek_pair_counts - agreements
    print(f'{total_pairs} {wikidata_pair_counts} {deepseek_pair_counts} {agreements} {found_in_wikidata_but_not_deepseek} {found_in_deepseek_but_not_wikidata}')
    return total_pairs, wikidata_pair_counts, deepseek_pair_counts, agreements, found_in_wikidata_but_not_deepseek, found_in_deepseek_but_not_wikidata
def result_venn_diagram(result):
    # library
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2

    # Use the venn2 function
    venn2(subsets=(result[4], result[5], result[3]), set_labels=('WikiData Pairs', 'Deepseek Pairs'))
    plt.show()


def main():
    doi_data_table = pd.read_csv(data_with_full_texts_csv, index_col=0)
    dois = doi_data_table['refDOI'].unique().tolist()
    results = get_counts_for_dois(dois)
    result_venn_diagram(results)
    results = get_counts_for_dois(dois, verbatim=False)
    result_venn_diagram(results)


if __name__ == '__main__':
    main()