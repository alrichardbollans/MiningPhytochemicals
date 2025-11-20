import re
from html import escape

from wcvpy.wcvp_name_matching import get_species_binomial_from_full_name, \
    get_species_from_full_name


def get_first_two_words(text):
    words = text.split()
    return ' '.join(words[:2])


# Function to highlight matches in text
def highlight_text(text, name, compound_name):
    # Escaping HTML characters in the text
    escaped_text = escape(text)
    if compound_name is not None:
        # Highlight matches for compound_name_list (blue)
        escaped_text = re.sub(
            re.escape(compound_name),
            f'<span style="color:blue; font-weight:bold;">{compound_name}</span>',
            escaped_text, flags=re.IGNORECASE
        )
        first_compounds_words = get_first_two_words(compound_name)
        # Highlight matches for compound_name_list (blue)
        escaped_text = re.sub(
            re.escape(first_compounds_words),
            f'<span style="color:blue; font-weight:bold;">{first_compounds_words}</span>',
            escaped_text, flags=re.IGNORECASE
        )
    if name is not None:
        # Highlight matches for name_list (red)
        escaped_text = re.sub(
            re.escape(name),
            f'<span style="color:red; font-weight:bold;">{name}</span>',
            escaped_text, flags=re.IGNORECASE
        )

        sp_binomial = get_species_binomial_from_full_name(name)
        escaped_text = re.sub(
            re.escape(sp_binomial),
            f'<span style="color:red; font-weight:bold;">{sp_binomial}</span>',
            escaped_text, flags=re.IGNORECASE
        )

        sp_epithet = get_species_from_full_name(name)
        escaped_text = re.sub(
            re.escape(sp_epithet),
            f'<span style="color:red; font-weight:bold;">{sp_epithet}</span>',
            escaped_text, flags=re.IGNORECASE
        )

    return escaped_text
