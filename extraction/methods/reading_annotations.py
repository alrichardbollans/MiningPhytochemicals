## these methods just provide some simple tidying of human and model annotations


import copy
import json
import os


TAXON_ENTITY_CLASSES = ['Scientific Plant Name']
MEDICINAL_CLASSES = ['Medical Condition', 'Medicinal Effect']
ENTITY_CLASSES = TAXON_ENTITY_CLASSES + MEDICINAL_CLASSES

MEDICINAL_RELATIONS = ['treats_medical_condition', 'has_medicinal_effect']


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


def clean_strings(given_str: str):
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


def standardise_NER_annotations(annotations: list):
    """
    Standardizes the NER annotations by cleaning the text strings.

    :param annotations: List of annotations dictionary.
    :return: None
    """

    for ann in annotations:
        ann['value']['text'] = clean_strings(ann['value']['text'])


def standardise_RE_annotations(annotations: list):
    """
    Standardizes the RE annotations by cleaning the text strings.

    :param annotations: A list of annotation dictionaries representing links between entities.
    :return: None
    """
    for ann in annotations:
        ann['from_entity']['value']['text'] = clean_strings(ann['from_entity']['value']['text'])
        ann['to_entity']['value']['text'] = clean_strings(ann['to_entity']['value']['text'])


def get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations(anns: list, check: bool = True, standardise_annotations:bool=True):
    """
    From a list of annotations (as given in annotations json), separate NER and RE annotations and return cleaned annotations (may be duplicates)
    """
    ner_annotations = [c for c in anns if c['type'] == 'labels']
    re_annotations = [c for c in anns if c['type'] == 'relation']

    ### Resolve entities in RE annotations as don't want to match based on entity number
    for re_annotation in re_annotations:
        from_entity = re_annotation['from_id']
        to_entity = re_annotation['to_id']
        for ner_annotation in ner_annotations:
            if ner_annotation['id'] == from_entity:
                re_annotation['from_entity'] = ner_annotation
            if ner_annotation['id'] == to_entity:
                re_annotation['to_entity'] = ner_annotation

    # separate annotations by class label
    separate_NER_annotations = []
    for ann in ner_annotations:

        for label in ann['value']['labels']:
            new_annotation = copy.deepcopy(ann)
            new_annotation['value']['label'] = label
            del new_annotation['value']['labels']
            separate_NER_annotations.append(new_annotation)

    if standardise_annotations:
        standardise_NER_annotations(separate_NER_annotations)

    separate_RE_annotations = []
    errors = []
    for ann in re_annotations:
        try:
            for label in ann['labels']:
                new_annotation = copy.deepcopy(ann)
                new_annotation['label'] = label
                del new_annotation['labels']
                separate_RE_annotations.append(new_annotation)
        except KeyError:
            errors.append(f'Relationship from "{ann["from_entity"]["value"]["text"]}" to "{ann["to_entity"]["value"]["text"]}" not given a label')
    if len(errors) > 0:
        raise KeyError(f"Errors found: {list(set(errors))}")

    if standardise_annotations:
        standardise_RE_annotations(separate_RE_annotations)
    if check:
        check_human_annotations(separate_NER_annotations, separate_RE_annotations)
    return separate_NER_annotations, separate_RE_annotations


def read_annotation_json(annotations_directory: str, corpus_id: str, chunk_id: str):
    """
    Reads annotation data from a JSON file.

    :param annotations_directory: The directory where the JSON file is located.
    :param corpus_id: The ID of the corpus.
    :param chunk_id: The ID of the chunk within the corpus.
    :return: A tuple containing two lists: separate_NER_annotations and separate_RE_annotations.
    """
    json_file = os.path.join(annotations_directory, f'task_for_labelstudio_{corpus_id}_chunk_{chunk_id}.json')
    with open(json_file) as f:
        d = json.load(f)
        # print(d)
    results = d[0]['predictions'][0]['result']
    return get_separate_NER_annotations_separate_RE_annotations_from_list_of_annotations(results)