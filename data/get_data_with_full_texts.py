import glob
import os

import pandas as pd

from data.get_wikidata import data_path, wikidata_plantae_reference_data_csv
from data.parse_refs import wikidatafulltext_dir, sanitise_doi

_data_with_full_texts_dir = os.path.join(data_path, 'wikidata', 'compound_data_with_full_texts')
data_with_full_texts_csv = os.path.join(data_path, 'wikidata', 'compound_data_with_full_texts', 'data_with_full_texts.csv')

test_data_csv = os.path.join(_data_with_full_texts_dir, 'test_data.csv')
validation_data_csv = os.path.join(_data_with_full_texts_dir, 'validation_data.csv')


def summarise_data_with_full_texts():
    df = pd.read_csv(wikidata_plantae_reference_data_csv, index_col=0)

    extension = 'txt'
    os.chdir(wikidatafulltext_dir)
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

    # For those already checked that I don't want to remove..
    # valdoi_data_table = pd.read_csv(validation_data_csv, index_col=0)
    # validation_dois = valdoi_data_table['refDOI'].unique().tolist()
    #
    # testdoi_data_table = pd.read_csv(test_data_csv, index_col=0)
    # test_dois = testdoi_data_table['refDOI'].unique().tolist()
    # [test_dois.remove(c) for c in already_checked_dois]

    ### Model split
    validation_dois = df_with_full_texts['refDOI'].unique().tolist()
    from sklearn.model_selection import train_test_split
    validation_dois, test_dois = train_test_split(validation_dois, test_size=0.997)

    assert len(validation_dois) == len(set(validation_dois))
    assert len(validation_dois) == 10

    validation_data = df_with_full_texts[df_with_full_texts['refDOI'].isin(validation_dois)]
    test_data = df_with_full_texts[df_with_full_texts['refDOI'].isin(test_dois)]
    validation_data.to_csv(validation_data_csv)
    validation_data.describe(include='all').to_csv(os.path.join(_data_with_full_texts_dir, 'validation_data_stats.csv'))
    test_data.to_csv(test_data_csv)
    test_data.describe(include='all').to_csv(os.path.join(_data_with_full_texts_dir, 'test_data_stats.csv'))


if __name__ == '__main__':
    summarise_data_with_full_texts()
