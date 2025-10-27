import json
import os
from typing import Optional, List, Any

import pandas as pd
from pydantic import BaseModel, Field



from extraction.methods.reading_annotations import clean_strings

repo_path = os.path.join(os.environ.get('KEWSCRATCHPATH'), 'MiningPhytochemicals')
data_path = os.path.join(repo_path, 'data')
base_text_path = os.path.join(data_path, 'fulltexts')




class Taxon(BaseModel):
    """Information about a plant or fungus."""

    # ^ Doc-string for the Taxon entity.
    # This doc-string is sent to the LLM as the description of the schema Taxon,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    scientific_name: Optional[str] = Field(default=None,
                                           description="The scientific name of the taxon, with scientific authority in the name if it appears in the text.")
    compounds: Optional[List[str]] = Field(
        default=None, description='Phytochemical compounds occurring in the taxon.'
    )


class TaxaData(BaseModel):
    """Extracted data about taxa."""

    # Creates a model so that we can extract multiple entities.
    taxa: Optional[List[Taxon]]

def deduplicate_and_standardise_output_taxa_lists(taxa: List[Taxon]) -> TaxaData:
    """ Clean strings, as in read_annotation_json and then deduplicate results"""
    unique_scientific_names = []
    for taxon in taxa:
        if taxon.scientific_name is not None:
            clean_name = clean_strings(taxon.scientific_name)
            if clean_name not in unique_scientific_names:
                unique_scientific_names.append(clean_name)

    new_taxa_list = []
    for name in unique_scientific_names:
        new_taxon = Taxon(scientific_name=name, compounds=[])
        for taxon in taxa:
            if clean_strings(taxon.scientific_name) == name:
                for condition in taxon.compounds or []:
                    if condition == condition and condition.lower() != 'null':
                        new_taxon.compounds.append(condition)

        if len(new_taxon.compounds) == 0:
            new_taxon.compounds = None
        else:
            cleaned_version = [clean_strings(c) for c in new_taxon.compounds]
            new_taxon.compounds = list(set(cleaned_version))

        new_taxa_list.append(new_taxon)
    return TaxaData(taxa=new_taxa_list)





def summarise_annotations(chunk_ids: list, out_path: str):
    from LLM_models.loading_files import get_txt_from_file

    number_of_chunks = len(chunk_ids)
    collected_taxa = []
    lone_taxa = []
    taxa_med_conds = []
    taxa_med_effects = []
    number_of_words = 0

    for chunk_id in chunk_ids:
        human_annotations = get_all_human_annotations_for_chunk_id(chunk_id, check=True)
        taxa = human_annotations.taxa
        for t in taxa:
            collected_taxa.append(t.scientific_name)
        number_of_words += len([i for i in get_txt_from_file(get_chunk_filepath_from_chunk_id(chunk_id)).split() if i.isalnum()])
        for t in taxa:
            if t.medicinal_effects is None and t.medical_conditions is None:
                lone_taxa.append(t.scientific_name)
            for m in t.medical_conditions or []:
                taxa_med_conds.append(f'{t.scientific_name}_{m}')
            for m in t.medicinal_effects or []:
                taxa_med_effects.append(f'{t.scientific_name}_{m}')
    collected_taxa = set(collected_taxa)
    lone_taxa = set(lone_taxa)
    taxa_med_conds = set(taxa_med_conds)
    taxa_med_effects = set(taxa_med_effects)

    raw = [number_of_chunks, len(collected_taxa), len(lone_taxa), len(taxa_med_conds), len(taxa_med_effects), number_of_words]
    mean = [x/number_of_chunks for x in raw]
    out_df = pd.DataFrame([raw, mean],
                          columns=['Number of Chunks', 'Taxa', 'Lone Taxa', 'Medical Conditions', 'Medicinal Effects','Number of Words'], index=['Total', 'Mean'])
    out_df.to_csv(out_path)


if __name__ == '__main__':
    check_all_human_annotations()
