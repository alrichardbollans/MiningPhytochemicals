import glob
import os

import pandas as pd

from data.get_compound_occurences import data_path, plantae_compounds_csv
from data.parse_refs import fulltext_dir, sanitise_doi

_data_with_full_texts_dir = os.path.join(data_path, 'compound_data_with_full_texts')
data_with_full_texts_csv = os.path.join(data_path, 'compound_data_with_full_texts', 'data_with_full_texts.csv')


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


if __name__ == '__main__':
    summarise_data_with_full_texts()
