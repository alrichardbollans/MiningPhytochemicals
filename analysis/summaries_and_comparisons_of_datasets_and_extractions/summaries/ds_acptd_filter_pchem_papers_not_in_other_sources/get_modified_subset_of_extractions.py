import json
import os
import pathlib

import pandas as pd
from phytochemMiner import TaxaData, Taxon
from phytochempy.compound_properties import simplify_inchi_key

from analysis.extraction_outputs.running_extraction import deepseek_jsons_path
from data.parse_refs import sanitise_doi


def filter_extractions_by_df(df:pd.DataFrame, output_folder):
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    # df = df.drop_duplicates(subset='pairs', keep='first')
    dois = df['refDOI'].unique().tolist()
    pairs_to_check = df['pairs'].unique().tolist()
    for doi in dois:
        sanitised_doi = sanitise_doi(doi)
        extraction_file = os.path.join(deepseek_jsons_path, sanitised_doi + '.json')

        json_dict = json.load(open(extraction_file, 'r'))
        deepseek_output = TaxaData.model_validate(json_dict)

        new_taxa_list = []

        for taxon in deepseek_output.taxa:
            new_taxon = Taxon(scientific_name=taxon.scientific_name, compounds=[])
            new_taxon.accepted_name = taxon.accepted_name
            new_taxon.accepted_species = taxon.accepted_species
            new_taxon.accepted_genus = taxon.accepted_genus
            inchi_keys = {}
            inchi_key_simps = {}
            compounds = []
            for compound in taxon.inchi_keys:
                inchi_simp = simplify_inchi_key(taxon.inchi_keys[compound])
                if taxon.accepted_name:
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
        with open(pathlib.Path(output_folder, sanitised_doi + '.json'), "w") as file_:
            json.dump(json_out, file_)


def main():
    df_to_use = pd.read_csv('filtered_occurrences.csv')
    filter_extractions_by_df(df_to_use, 'relevant_jsons')


if __name__ == '__main__':
    main()
