import os

import pandas as pd
from tqdm import tqdm

from data.get_colombian_data import get_sanitised_dois_for_colombian_papers
from data.get_data_with_full_texts import validation_data_csv
from data.get_papers_with_no_hits import get_sanitised_dois_for_papers
from data.parse_refs import wikidatafulltext_dir, sanitise_doi
from phytochemMiner import get_phytochem_model, run_phytochem_model

repo_path = os.path.join(os.environ.get('KEWSCRATCHPATH'), 'MiningPhytochemicals')
deepseek_jsons_path = os.path.join(repo_path, 'analysis', 'extraction', 'deepseek_jsons')


def main():
    model, limit = get_phytochem_model(dotenv_path='.env')

    ## Validation data examples
    doi_data_table = pd.read_csv(validation_data_csv, index_col=0)
    for doi in doi_data_table['refDOI'].unique().tolist():
        print('###########')
        print(doi)
        sanitised_doi = sanitise_doi(doi)
        fulltextpath = os.path.join(wikidatafulltext_dir, f'{sanitised_doi}.txt')
        result_ = run_phytochem_model(model, fulltextpath,
                                      limit,
                                      json_dump=os.path.join(deepseek_jsons_path, sanitised_doi + '.json'), rerun=False,
                                      rerun_inchi_resolution=False)

    ### Negative examples
    random_txt_dir, result = get_sanitised_dois_for_papers('random papers')
    for sanitised_doi in result:
        print('###########')
        print(sanitised_doi)
        fulltextpath = os.path.join(random_txt_dir, f'{sanitised_doi}.txt')
        result_ = run_phytochem_model(model, fulltextpath,
                                      limit,
                                      json_dump=os.path.join(deepseek_jsons_path, sanitised_doi + '.json'), rerun=False,
                                      rerun_inchi_resolution=False)

    medplant_txt_dir, result = get_sanitised_dois_for_papers('medplant papers')
    for sanitised_doi in result:
        print('###########')
        print(sanitised_doi)
        fulltextpath = os.path.join(medplant_txt_dir, f'{sanitised_doi}.txt')
        result_ = run_phytochem_model(model, fulltextpath,
                                      limit,
                                      json_dump=os.path.join(deepseek_jsons_path, sanitised_doi + '.json'), rerun=False,
                                      rerun_inchi_resolution=False)

    ## Phytochem paper examples
    phytochem_txt_dir, result = get_sanitised_dois_for_papers('phytochemistry papers')
    for i in tqdm(range(len(result))):
        sanitised_doi = result[i]
        fulltextpath = os.path.join(phytochem_txt_dir, f'{sanitised_doi}.txt')
        result_ = run_phytochem_model(model, fulltextpath,
                                      limit,
                                      json_dump=os.path.join(deepseek_jsons_path, sanitised_doi + '.json'), rerun=False,
                                      rerun_inchi_resolution=False)

    ### colombian paper examples
    colombian_dois = get_sanitised_dois_for_colombian_papers()
    for sanitised_doi in colombian_dois:
        fulltextpath = os.path.join(colombian_dois[sanitised_doi], f'{sanitised_doi}.txt')
        result_ = run_phytochem_model(model, fulltextpath,
                                      limit,
                                      json_dump=os.path.join(deepseek_jsons_path, sanitised_doi + '.json'), rerun=False,
                                      rerun_inchi_resolution=False)


if __name__ == '__main__':
    main()
