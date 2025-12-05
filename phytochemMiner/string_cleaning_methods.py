def remove_double_spaces_and_break_characters(given_text: str) -> str:
    '''
    This will simplify a text by removing double spaces and all whitespace characters (e.g. space, tab, newline, return, formfeed).
    See https://stackoverflow.com/a/1546251/8633026
    This may or may not be desirable and should only be used at the end of preprocessing as it removes important characters like \n.
    :param given_text:
    :return:
    '''
    if given_text:
        return " ".join(given_text.split())
    else:
        return given_text


def leading_trailing_whitespace(given_str: str):
    """
    Remove leading and trailing whitespace from a given string.

    :param given_str: The input string.
    :return: The input string with leading and trailing whitespace removed.
    """
    try:
        if given_str != given_str or given_str is None:
            return given_str
        else:
            stripped = given_str.strip()
            return stripped
    except AttributeError:
        return given_str


def leading_trailing_punctuation(given_str: str):
    """
    :param given_str: The input string which may contain leading and trailing punctuation.
    :return: The input string with leading and trailing punctuation removed.

    """
    try:
        if given_str != given_str or given_str is None:
            return given_str
        else:
            stripped = given_str.strip('!"#$%&\'()*,-./:;<=>?@[\\]^_`{|}~')
            return stripped
    except AttributeError:
        return given_str


def lowercase(given_str: str):
    """
    Convert the given string to lowercase.

    :param given_str: The string to be converted to lowercase.
    :return: The lowercase version of the given string.
    """
    try:
        if given_str != given_str or given_str is None:
            return given_str
        else:
            return given_str.lower()
    except AttributeError:
        return given_str


def clean_taxon_strings(given_str: str):
    """
    Clean the given string by removing leading/trailing whitespace,
    leading/trailing punctuation, and converting all characters to lowercase.

    Also remove double spaces and break characters, as with files that are passed to LLM models.

    A clean string should be retrievable from the original text when all lower case.

    :param given_str: The string to be cleaned.
    :return: The cleaned string.
    """
    low = lowercase(given_str)
    while (leading_trailing_whitespace(low) != low) or (leading_trailing_punctuation(low) != low):
        low = leading_trailing_whitespace(low)
        low = leading_trailing_punctuation(low)
    return remove_double_spaces_and_break_characters(low)


def clean_compound_strings(given_str: str):
    """
    Clean the given string by removing leading/trailing whitespace and converting all characters to lowercase.

    Also remove double spaces and break characters, as with files that are passed to LLM models.

    A clean string should be retrievable from the original text when all lower case.

    :param given_str: The string to be cleaned.
    :return: The cleaned string.
    """
    low = lowercase(given_str)
    while (leading_trailing_whitespace(low) != low):
        low = leading_trailing_whitespace(low)
    return remove_double_spaces_and_break_characters(low)
