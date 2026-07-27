# All pathways, superclasses and classes from:
# Kim, H.W. et al. (2021) ‘NPClassifier: A Deep Neural Network-Based Structural Classification Tool for Natural Products’,
# Journal of Natural Products, 84(11), pp. 2795–2807. Available at: https://doi.org/10.1021/acs.jnatprod.1c00399.

# File from https://github.com/mwang87/NP-Classifier/blob/78e52f1d27484841b303417dc2847430036092d6/Classifier/dict/index_v1.json

import json
import os

import pandas as pd
from phytochemMiner import get_classes


def remove_classes(folder, classes_to_remove):
    occurrences = pd.read_csv(os.path.join(folder, 'occurrences.csv'), index_col=0)
    occurrences['pairs'] = occurrences['accepted_name'] + '_' + occurrences['InChIKey_simp']
    filtered_occurrences = occurrences[~occurrences['extracted_compound_name'].str.lower().isin(classes_to_remove)]
    filtered_occurrences.to_csv(os.path.join(folder, 'filtered_occurrences.csv'))
    filtered_occurrences.describe(include='all').to_csv(os.path.join(folder, 'filtered_occurrences_summary.csv'))


def main():
    all_classes_lower = get_classes()

    remove_classes('summaries/deepseek_after_accepted_filter_phytochem_papers', all_classes_lower)
    remove_classes('summaries/deepseek_after_accepted_filter_phytochem_papers_not_in_other_sources', all_classes_lower)


if __name__ == '__main__':
    main()
