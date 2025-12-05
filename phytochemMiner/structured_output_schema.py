from typing import Optional, List

from pydantic import BaseModel, Field, Extra

from phytochemMiner import clean_taxon_strings, clean_compound_strings


class Taxon(BaseModel, extra=Extra.allow):
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


class TaxaData(BaseModel, extra=Extra.allow):
    """Extracted data about taxa."""

    # Creates a model so that we can extract multiple entities.
    taxa: Optional[List[Taxon]]


def deduplicate_and_standardise_output_taxa_lists(taxa: List[Taxon], ) -> TaxaData:
    """ Clean strings, as in read_annotation_json and then deduplicate results"""
    unique_scientific_names = []
    for taxon in taxa:
        if taxon.scientific_name is not None:
            clean_name = clean_taxon_strings(taxon.scientific_name)
            if clean_name not in unique_scientific_names:
                unique_scientific_names.append(clean_name)

    new_taxa_list = []
    for name in unique_scientific_names:
        new_taxon = Taxon(scientific_name=name, compounds=[])
        for taxon in taxa:
            if clean_taxon_strings(taxon.scientific_name) == name:
                for condition in taxon.compounds or []:
                    if condition == condition and condition.lower() != 'null':
                        new_taxon.compounds.append(condition)

        if len(new_taxon.compounds) == 0:
            new_taxon.compounds = None
        else:
            cleaned_version = [clean_compound_strings(c) for c in new_taxon.compounds]
            new_taxon.compounds = list(set(cleaned_version))

        new_taxa_list.append(new_taxon)
    return TaxaData(taxa=new_taxa_list)
