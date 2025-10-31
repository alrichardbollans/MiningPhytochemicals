import glob
import os

import pandas as pd

from data.get_compound_occurences import data_path, plantae_compounds_csv
from data.parse_refs import fulltext_dir, sanitise_doi

_data_with_full_texts_dir = os.path.join(data_path, 'compound_data_with_full_texts')
data_with_full_texts_csv = os.path.join(data_path, 'compound_data_with_full_texts', 'data_with_full_texts.csv')

test_data_csv = os.path.join(_data_with_full_texts_dir, 'test_data.csv')
validation_data_csv = os.path.join(_data_with_full_texts_dir, 'validation_data.csv')


def summarise_data_with_full_texts():
    df = pd.read_csv(plantae_compounds_csv, index_col=0)

    extension = 'txt'
    os.chdir(fulltext_dir)
    result = glob.glob('*.{}'.format(extension))

    result = [c[:-4] for c in result[:]]

    # check there are no collisions from doi sanitisation
    df['sanitised_dois'] = df['refDOI'].apply(sanitise_doi)
    grouped = df.groupby('sanitised_dois')['refDOI'].nunique()
    assert grouped.max() == 1

    df_with_full_texts = df[df['sanitised_dois'].isin(result)]
    assert len(df_with_full_texts['refDOI'].unique().tolist()) < len(result)
    df_with_full_texts.to_csv(data_with_full_texts_csv)
    df_with_full_texts.describe(include='all').to_csv(
        os.path.join(_data_with_full_texts_dir, 'data_with_full_texts_summary.csv'))

    ### Model split
    validation_dois = df_with_full_texts['refDOI'].unique().tolist()
    from sklearn.model_selection import train_test_split
    validation_dois, test_dois = train_test_split(validation_dois, test_size=0.997)

    already_checked_dois = ['10.3389/FPLS.2012.00283', '10.1590/S0100-40422001000100006',
                            '10.1093/ECAM/NEQ006', '10.1080/14786410903052399',
                            '10.3892/IJO.2015.2946', '10.1186/1742-6405-2-12',
                            '10.1007/S11306-009-0195-X', '10.1016/J.PHYTOCHEM.2010.06.020'
                        ]
    validation_dois += already_checked_dois
    assert len(validation_dois) == len(set(validation_dois))
    assert len(validation_dois) == 10
    [test_dois.remove(c) for c in already_checked_dois]

    validation_data = df_with_full_texts[df_with_full_texts['refDOI'].isin(validation_dois)]
    test_data = df_with_full_texts[df_with_full_texts['refDOI'].isin(test_dois)]
    validation_data.to_csv(validation_data_csv)
    validation_data.describe(include='all').to_csv(os.path.join(_data_with_full_texts_dir, 'validation_data_stats.csv'))
    test_data.to_csv(test_data_csv)
    test_data.describe(include='all').to_csv(os.path.join(_data_with_full_texts_dir, 'test_data_stats.csv'))


if __name__ == '__main__':
    summarise_data_with_full_texts()
