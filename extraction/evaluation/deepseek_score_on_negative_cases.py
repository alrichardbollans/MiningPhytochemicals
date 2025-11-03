import os
import pickle

from django.template.defaulttags import verbatim

from data.get_papers_with_no_hits import get_sanitised_dois_for_random_papers
from extraction.methods.get_agreements_and_disagreements import get_verbatim_matches
from extraction.methods.running_models import deepseek_pkls_path


def main():
    random_txt_dir, result = get_sanitised_dois_for_random_papers()
    errors = []
    for sanitised_doi in result:
        deepseek_output = pickle.load(open(os.path.join(deepseek_pkls_path, sanitised_doi + '.pkl'), 'rb'))
        verbatim_cases = get_verbatim_matches(deepseek_output,deepseek_output)
        deduplicated_output = verbatim_cases[0]
        for taxon in deduplicated_output.taxa:
            for compound in taxon.compounds:
                errors.append((taxon.scientific_name, compound))
    print(errors)

if __name__ == '__main__':
    main()