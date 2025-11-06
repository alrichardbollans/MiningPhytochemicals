import os
import pickle

from phytochempy.knapsack_searches import get_knapsack_data, get_knapsack_compounds_in_family
from tqdm import tqdm

from data.get_compound_occurences import family_pkl_file, data_path

knapsack_data_path = os.path.join(data_path, 'knapsack_data')
def main():
    _temp_path = os.path.join(knapsack_data_path, 'temp')

    fam_dict = pickle.load(open(family_pkl_file, 'rb'))
    families_of_interest = list(set(fam_dict.keys()))
    for i in tqdm(range(len(families_of_interest)), desc=f"Searching families…"):
        family = families_of_interest[i]
        get_knapsack_compounds_in_family(family, os.path.join(_temp_path, f'{family}.csv'))


if __name__ == '__main__':
    main()