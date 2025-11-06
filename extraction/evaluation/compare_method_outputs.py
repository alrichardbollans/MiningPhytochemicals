import os
import pickle

import pandas as pd

from data.get_compound_occurences import inchi_translation_cache
from data.get_data_with_full_texts import data_with_full_texts_csv, validation_data_csv, test_data_csv
from extraction.methods.get_agreements_and_disagreements import check_records_for_doi
from extraction.methods.structured_output_schema import TaxaData

pkled_inchi_translation_result = pickle.load(open(inchi_translation_cache, 'rb'))


def get_wikidata_examples_to_check(dois: list):
    wikidata_potentially_bad_examples = []
    for doi in dois:
        verbatim_results, accepted_results = check_records_for_doi(doi)
        for taxa in accepted_results[3].taxa:
            for compound in taxa.inchi_key_simps:
                wikidata_potentially_bad_examples.append([taxa.accepted_name, str(taxa.matched_names),compound,taxa.inchi_key_simps[compound], doi, 'wikidata'])
    out_df = pd.DataFrame(wikidata_potentially_bad_examples, columns=['accepted_name', 'matched_names', 'compound', 'inchi_key_simps','doi', 'source'])
    out_df.to_csv(os.path.join('outputs', 'wikidata_deepseek_comparison', 'wikidata_potentially_bad_examples.csv'))


def count_unresolved_compound_names(taxadata: TaxaData):
    resolved = 0
    unresolved = 0
    for taxon in taxadata.taxa:
        for name in taxon.compounds:
            if name in pkled_inchi_translation_result and pkled_inchi_translation_result[name] is not None and pkled_inchi_translation_result[name] != '':
                resolved += 1
            else:
                unresolved += 1
    return resolved, unresolved


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
            unique_simps = set(taxon.inchi_key_simps.values())
            counts += len(unique_simps)
    return counts


def get_counts_for_dois(dois: list, verbatim=True):
    wikidata_pair_counts = 0
    wikidata_name_counts = 0
    deepseek_pair_counts = 0
    deepseek_resolved_compound_names = 0
    deepseek_unresolved_compound_names = 0
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

        resolved_compound_names, unresolved_compound_names = count_unresolved_compound_names(results[1])
        deepseek_resolved_compound_names += resolved_compound_names
        deepseek_unresolved_compound_names += unresolved_compound_names

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
            'deepseek_name_counts': deepseek_name_counts,
            'deepseek_resolved_compound_names': deepseek_resolved_compound_names,
            'deepseek_unresolved_compound_names': deepseek_unresolved_compound_names}


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

    get_wikidata_examples_to_check(dois)

    results = get_counts_for_dois(dois)
    pd.DataFrame.from_dict(results, orient='index').to_csv(
        os.path.join(outpath, 'verbatim_validation_data_counts.csv'))
    result_venn_diagram(results, os.path.join(outpath, 'verbatim_validation_data_venn.jpg'))
    results = get_counts_for_dois(dois, verbatim=False)
    pd.DataFrame.from_dict(results, orient='index').to_csv(os.path.join(outpath, 'resolved_validation_data_counts.csv'))
    result_venn_diagram(results, os.path.join(outpath, 'resolved_validation_data_venn.jpg'))

    ## Test Data
    doi_data_table = pd.read_csv(test_data_csv, index_col=0)
    dois = doi_data_table['refDOI'].unique().tolist()
    results = get_counts_for_dois(dois)
    pd.DataFrame.from_dict(results, orient='index').to_csv(
        os.path.join(outpath, 'verbatim_test_data_counts.csv'))
    result_venn_diagram(results, os.path.join(outpath, 'verbatim_test_data_venn.jpg'))
    results = get_counts_for_dois(dois, verbatim=False)
    pd.DataFrame.from_dict(results, orient='index').to_csv(
        os.path.join(outpath, 'resolved_test_data_counts.csv'))
    result_venn_diagram(results, os.path.join(outpath, 'resolved_test_data_venn.jpg'))


if __name__ == '__main__':
    main()
