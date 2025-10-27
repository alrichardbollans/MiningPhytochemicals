import os

import pandas as pd

from data.get_compound_occurences import data_path, plantae_compounds_csv

data_summary_path = os.path.join(data_path, 'compound_data_with_full_texts')


def summarise_data_with_full_texts():
    df = pd.read_csv(plantae_compounds_csv, index_col=0)

if __name__ == '__main__':
    pass