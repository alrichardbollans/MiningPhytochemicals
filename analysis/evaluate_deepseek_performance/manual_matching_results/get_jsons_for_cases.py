import json
import os
import pathlib

import pandas as pd
from phytochemMiner import TaxaData, Taxon, get_classes
from phytochempy.compound_properties import simplify_inchi_key

from analysis.extraction_outputs.running_extraction import deepseek_jsons_path
from data.get_colombian_data import get_sanitised_dois_for_colombian_papers
from data.get_data_with_full_texts import validation_data_csv
from data.get_papers_with_no_hits import get_sanitised_dois_for_papers
from data.parse_refs import sanitise_doi
import shutil


def main():
    print(f'cwd: {os.getcwd()}')
    extracted_jsons_folder = 'extracted_jsons'
    pathlib.Path(extracted_jsons_folder).mkdir(parents=True, exist_ok=True)
    # pathlib.Path(os.path.join(extracted_jsons_folder, 'validation cases')).mkdir(parents=True, exist_ok=True)
    # # Validation cases
    # doi_data_table = pd.read_csv(validation_data_csv, index_col=0)
    # for doi in doi_data_table['refDOI'].unique().tolist():
    #     print('###########')
    #     sanitised_doi = sanitise_doi(doi)
    #     print(sanitised_doi)
    #     json_file = os.path.join(deepseek_jsons_path, sanitised_doi + '.json')
    #     shutil.copyfile(json_file, os.path.join(extracted_jsons_folder, 'validation cases', sanitised_doi + '.json'))
    #
    # colombian_dois = get_sanitised_dois_for_colombian_papers()
    # pathlib.Path(os.path.join(extracted_jsons_folder, 'colombian papers')).mkdir(parents=True, exist_ok=True)
    # pathlib.Path(os.path.join('manual results after accepted filter', 'colombian papers')).mkdir(parents=True, exist_ok=True)
    # for sanitised_doi in colombian_dois:
    #     json_file = os.path.join(deepseek_jsons_path, sanitised_doi + '.json')
    #     shutil.copyfile(json_file, os.path.join(extracted_jsons_folder, 'colombian papers', sanitised_doi + '.json'))

    # text_dir, phytochemistry_dois = get_sanitised_dois_for_papers('phytochemistry papers')
    # pathlib.Path(os.path.join(extracted_jsons_folder, 'phytochemistry papers')).mkdir(parents=True, exist_ok=True)
    # pathlib.Path(os.path.join('manual results after accepted filter', 'phytochemistry papers')).mkdir(parents=True, exist_ok=True)
    # for sanitised_doi in phytochemistry_dois:
    #     json_file = os.path.join(deepseek_jsons_path, sanitised_doi + '.json')
    #     shutil.copyfile(json_file, os.path.join(extracted_jsons_folder, 'phytochemistry papers', sanitised_doi + '.json'))
    all_classes_lower = get_classes()

    pairs_to_check_df = pd.read_csv(pathlib.Path(
        '..','..','summaries_and_comparisons_of_datasets_and_extractions','summaries','ds_acptd_filter_pchem_papers_not_in_other_sources','occurrences.csv'))
    pairs_to_check_df['pairs'] = pairs_to_check_df['accepted_name'] + '_' + pairs_to_check_df['InChIKey_simp']
    pairs_to_check = pairs_to_check_df['pairs'].unique().tolist()
    text_dir, phytochemistry_dois = get_sanitised_dois_for_papers('phytochemistry papers')
    out_folder_name = 'pchem hits not in WD or KN'
    pathlib.Path(os.path.join(extracted_jsons_folder, out_folder_name)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join('manual results after accepted filter', out_folder_name)).mkdir(parents=True, exist_ok=True)
    for sanitised_doi in phytochemistry_dois:
        in_json_file = os.path.join(deepseek_jsons_path, sanitised_doi + '.json')
        json_object = json.load(open(in_json_file, 'r'))
        deepseek_output = TaxaData.model_validate(json_object)

        new_taxa_list = []

        for taxon in deepseek_output.taxa:
            new_taxon = Taxon(scientific_name=taxon.scientific_name, compounds=[])
            new_taxon.accepted_name = taxon.accepted_name
            new_taxon.accepted_species = taxon.accepted_species
            new_taxon.accepted_genus = taxon.accepted_genus
            try:
                new_taxon.text = taxon.text
            except AttributeError:
                pass
            inchi_keys = {}
            inchi_key_simps = {}
            compounds = []
            for compound in taxon.inchi_keys:
                inchi_simp = simplify_inchi_key(taxon.inchi_keys[compound])
                if taxon.accepted_name and compound.lower() not in all_classes_lower:
                    if (taxon.accepted_name + '_' + inchi_simp) in pairs_to_check:
                        inchi_keys[compound] = taxon.inchi_keys[compound]
                        inchi_key_simps[compound] = inchi_simp
                        compounds.append(compound)
            new_taxon.inchi_keys = inchi_keys
            new_taxon.inchi_key_simps = inchi_key_simps
            new_taxon.compounds = compounds
            if len(compounds) > 0:
                new_taxa_list.append(new_taxon)

        new_taxa_data = TaxaData(taxa=new_taxa_list)
        new_taxa_data.text = deepseek_output.text
        json_out = new_taxa_data.model_dump(mode="json")
        with open(pathlib.Path(extracted_jsons_folder, out_folder_name, sanitised_doi + '_not_in_WD_KN.json'), "w") as file_:
            json.dump(json_out, file_)


if __name__ == '__main__':
    main()
