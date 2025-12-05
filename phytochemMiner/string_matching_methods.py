from wcvpy.wcvp_download import hybrid_characters
from wcvpy.wcvp_name_matching import get_species_binomial_from_full_name

from phytochemMiner import clean_taxon_strings, clean_compound_strings


def abbreviate_sci_name(name1: str) -> str:
    """
    Return given name with first word abbreviated, if there are multiple words.
    :param name1:
    :return:
    """

    words = name1.split()
    if len(words) < 2:
        return name1
    else:
        if words[0] in hybrid_characters:
            if len(words) < 3:
                return name1
            else:
                words[1] = words[1][0] + '.'
        else:
            words[0] = words[0][0] + '.'
        return ' '.join(words)


def precise_taxon_name_match(name1: str, name2: str):
    cleaned_name2 = get_species_binomial_from_full_name(clean_taxon_strings(name2))

    cleaned_name1 = get_species_binomial_from_full_name(clean_taxon_strings(name1))
    if cleaned_name1 == cleaned_name2 or "".join(cleaned_name1.split()) == "".join(cleaned_name2.split()):
        return True
    else:

        return False


def check_organism_names_match(name1: str, name2: str):
    """
    :param name1: The first name to compare.
    :param name2: The second name to compare.
    :return: True if name1 is an exact match or an abbreviation of name2, or if name2 is an abbreviation of name1. False otherwise.
    """
    if name1 is None or name2 is None or name1 == "" or name2 == "":
        raise ValueError("Both names must be provided.")

    if precise_taxon_name_match(name1, name2):
        return True
    if precise_taxon_name_match(abbreviate_sci_name(name1), name2) or precise_taxon_name_match(
            abbreviate_sci_name(name2), name1):
        return True
    else:
        return False


def check_compound_names_match(name1: str, name2: str):
    cleaned_name2 = clean_compound_strings(name2)

    cleaned_name1 = clean_compound_strings(name1)
    if cleaned_name1 == cleaned_name2 or "".join(cleaned_name1.split()) == "".join(cleaned_name2.split()):
        return True
    else:

        return False
