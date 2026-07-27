# All pathways, superclasses and classes from:
# Kim, H.W. et al. (2021) ‘NPClassifier: A Deep Neural Network-Based Structural Classification Tool for Natural Products’,
# Journal of Natural Products, 84(11), pp. 2795–2807. Available at: https://doi.org/10.1021/acs.jnatprod.1c00399.

# File from https://github.com/mwang87/NP-Classifier/blob/78e52f1d27484841b303417dc2847430036092d6/Classifier/dict/index_v1.json

import json
import os

import pandas as pd


def remove_classes(folder, classes_to_remove):
    occurrences = pd.read_csv(os.path.join(folder, 'occurrences.csv'), index_col=0)
    occurrences['pairs'] = occurrences['accepted_name'] + '_' + occurrences['InChIKey_simp']
    filtered_occurrences = occurrences[~occurrences['extracted_compound_name'].str.lower().isin(classes_to_remove)]
    filtered_occurrences.to_csv(os.path.join(folder, 'filtered_occurrences.csv'))
    filtered_occurrences.describe(include='all').to_csv(os.path.join(folder, 'filtered_occurrences_summary.csv'))


def get_classes():
    class_json = json.load(open('index_v1.json'))
    classes = set(class_json['Class'].keys())
    superclasses = set(class_json['Superclass'].keys())
    pathways = set(class_json['Pathway'].keys())

    all_classes = classes | superclasses | pathways

    ## Add some classes manually based on those found in cololmbia paper
    all_classes.add('saponins')
    all_classes.add('tannins')
    all_classes.add('flavonoids')
    all_classes.add('steroids')
    all_classes.add('non-reducing sugars')
    all_classes.add('hemolytic saponins')
    all_classes.add('organic acids')
    all_classes.add('catechins')
    all_classes.add('depsides and depsidones')
    all_classes.add('double olefins')
    all_classes.add('reducing sugars')
    all_classes.add('resins')
    all_classes.add('sesquiterpenolactones')
    all_classes.add('phenols')
    all_classes.add('purines')
    all_classes.add('polyphenols')
    all_classes.add('triterpenes')

    all_classes_lower = []

    # lower case everything in all classes
    for class_ in all_classes:
        all_classes_lower.append(class_.lower())

    assert 'alkaloids' in all_classes_lower
    assert 'saponins' in all_classes_lower

    return all_classes_lower


def main():
    all_classes_lower = get_classes()

    remove_classes('summaries/deepseek_after_accepted_filter_phytochem_papers', all_classes_lower)
    remove_classes('summaries/deepseek_after_accepted_filter_phytochem_papers_not_in_other_sources', all_classes_lower)


if __name__ == '__main__':
    main()
