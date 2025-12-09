import os
import pathlib

import pandas as pd

from analysis.extraction.running_extraction import deepseek_jsons_path
from data.get_colombian_data import get_sanitised_dois_for_colombian_papers
from data.get_data_with_full_texts import validation_data_csv
from data.parse_refs import sanitise_doi
import shutil


def main():
    extracted_jsons_folder = 'extracted_jsons'
    pathlib.Path(extracted_jsons_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(extracted_jsons_folder, 'validation cases')).mkdir(parents=True, exist_ok=True)
    # Validation cases
    doi_data_table = pd.read_csv(validation_data_csv, index_col=0)
    for doi in doi_data_table['refDOI'].unique().tolist():
        print('###########')
        sanitised_doi = sanitise_doi(doi)
        print(sanitised_doi)
        json_file = os.path.join(deepseek_jsons_path, sanitised_doi + '.json')
        shutil.copyfile(json_file, os.path.join(extracted_jsons_folder, 'validation cases', sanitised_doi + '.json'))

    colombian_dois = get_sanitised_dois_for_colombian_papers()
    pathlib.Path(os.path.join(extracted_jsons_folder, 'colombian papers')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join('manual results', 'colombian papers')).mkdir(parents=True, exist_ok=True)
    for sanitised_doi in colombian_dois:
        json_file = os.path.join(deepseek_jsons_path, sanitised_doi + '.json')
        shutil.copyfile(json_file, os.path.join(extracted_jsons_folder, 'colombian papers', sanitised_doi + '.json'))


if __name__ == '__main__':
    main()
