import os
import pathlib

import pandas as pd

from data.get_colombian_data import get_sanitised_dois_for_colombian_papers
from data.get_data_with_full_texts import validation_data_csv
from data.parse_refs import sanitise_doi
from extraction.methods.running_models import deepseek_pkls_path
import shutil


def main():
    extracted_pkl_folder = 'extracted_pkls'
    pathlib.Path(extracted_pkl_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(extracted_pkl_folder, 'validation cases')).mkdir(parents=True, exist_ok=True)
    # Validation cases
    doi_data_table = pd.read_csv(validation_data_csv, index_col=0)
    for doi in doi_data_table['refDOI'].unique().tolist():
        print('###########')
        sanitised_doi = sanitise_doi(doi)
        print(sanitised_doi)
        pkl = os.path.join(deepseek_pkls_path, sanitised_doi + '.pkl')
        shutil.copyfile(pkl, os.path.join(extracted_pkl_folder, 'validation cases', sanitised_doi + '.pkl'))

    colombian_dois = get_sanitised_dois_for_colombian_papers()
    pathlib.Path(os.path.join(extracted_pkl_folder, 'colombian papers')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join('manual results', 'colombian papers')).mkdir(parents=True, exist_ok=True)
    for sanitised_doi in colombian_dois:
        pkl = os.path.join(deepseek_pkls_path, sanitised_doi + '.pkl')
        shutil.copyfile(pkl, os.path.join(extracted_pkl_folder, 'colombian papers', sanitised_doi + '.pkl'))


if __name__ == '__main__':
    main()
