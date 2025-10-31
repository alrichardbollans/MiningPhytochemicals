import os
import pickle

import langchain_core
import pandas as pd
import pydantic_core

from data.get_data_with_full_texts import validation_data_csv
from data.parse_refs import fulltext_dir, sanitise_doi
from extraction.methods.loading_files import read_file_and_chunk
from extraction.methods.prompting import standard_prompt
from extraction.methods.structured_output_schema import TaxaData, deduplicate_and_standardise_output_taxa_lists

repo_path = os.path.join(os.environ.get('KEWSCRATCHPATH'), 'MiningPhytochemicals')
deepseek_pkls_path = os.path.join(repo_path, 'extraction', 'deepseek_pkls')


def query_a_model(model, text_file: str, context_window: int, pkl_dump: str = None,
                  single_chunk: bool = True, rerun=True) -> TaxaData:
    if not rerun and os.path.exists(pkl_dump):
        with open(pkl_dump, "rb") as file_:
            return pickle.load(file_)

    if not single_chunk:
        raise NotImplementedError
    text_chunks = read_file_and_chunk(text_file, context_window)
    if single_chunk:
        # For most of analysis, will be testing on single chunks as this is how we've annotated them.
        # In this instance, the chunks should fit in the context window
        assert len(text_chunks) == 1
    # A few different methods, depending on the specific model are used to get a structured output
    # and this is handled by with_structured_output. See https://python.langchain.com/docs/how_to/structured_output/
    extractor = standard_prompt | model.with_structured_output(schema=TaxaData, include_raw=False)
    try:

        extractions = extractor.batch(
            [{"text": text} for text in text_chunks],
            {"max_concurrency": 1},
            # limit the concurrency by passing max concurrency! Otherwise Requests rate limit exceeded
        )
    except (langchain_core.exceptions.OutputParserException, pydantic_core._pydantic_core.ValidationError) as e:
        raise NotImplementedError
        # When there is too much info extracted the extractor can't parse the output json, so make chunks smaller.
        # This can also happen because of limits on model max output tokens
        print(f'Warning: reducing size of chunk as output json is too large to parse. For file {text_file}')

        new_chunks = split_text_chunks(text_chunks)
        print(f'Length of old chunk: {len(text_chunks[0])}')

        extractions = []
        for text in new_chunks:
            try:
                chunk_output = extractor.invoke({"text": text})
                extractions.append(chunk_output)
            except Exception as e:
                more_chunks = split_text_chunks([text])
                for more_text in more_chunks:
                    try:
                        chunk_output = extractor.invoke({"text": more_text})
                        extractions.append(chunk_output)
                    except Exception as e:
                        # print(f'Unknown error "{e}" for text with length {len(more_text)}: {more_text}')
                        even_more_chunks = split_text_chunks([more_text])
                        for even_more_text in even_more_chunks:
                            try:
                                chunk_output = extractor.invoke({"text": even_more_text})
                                extractions.append(chunk_output)
                            except Exception as e:
                                print(
                                    f'Unknown error "{e}" for text with length {len(even_more_text)}: {even_more_text}')

    output = []

    for extraction in extractions:
        if extraction.taxa is not None:
            output.extend(extraction.taxa)

    deduplicated_extractions = deduplicate_and_standardise_output_taxa_lists(output)
    add_all_extra_info_to_output(deduplicated_extractions)
    if pkl_dump:
        with open(pkl_dump, "wb") as file_:
            pickle.dump(deduplicated_extractions, file_)

    return deduplicated_extractions


def get_input_size_limit(total_context_window_k: int):
    # Output tokens so far is a tiny fraction, so allow 5% of context window for output
    out_units = total_context_window_k * 1000
    input_size = int(out_units * 0.95)
    return input_size


def setup_models():
    # Get API keys
    from dotenv import load_dotenv

    load_dotenv()
    out = {}

    hparams = {'temperature': 0}

    # DeepSeek V3
    # Created 30/12/2024
    # Max tokens 128k
    # Input/Output: $0.28/0.42/1M tokens
    # https://api-docs.deepseek.com/quick_start/pricing/

    # DeepSeek-R1, specified via model="deepseek-reasoner", does not support tool calling or structured output.
    # Those features are supported by DeepSeek-V3 (specified via model="deepseek-chat").

    from langchain_deepseek import ChatDeepSeek
    model6 = ChatDeepSeek(
        model="deepseek-chat", **hparams)
    out['deepseek-chat'] = [model6, get_input_size_limit(128)]

    return out


def main():
    models = setup_models()

    example_model_name = 'deepseek-chat'
    doi_data_table = pd.read_csv(validation_data_csv, index_col=0)
    for doi in doi_data_table['refDOI'].unique().tolist():
        print('###########')
        print(doi)
        sanitised_doi = sanitise_doi(doi)
        fulltextpath = os.path.join(fulltext_dir, f'{sanitised_doi}.txt')
        result_ = query_a_model(models[example_model_name][0], fulltextpath,
                                models[example_model_name][1],
                                pkl_dump=os.path.join(deepseek_pkls_path, sanitised_doi + '.pkl'), rerun=False)

        print(result_)
    #
    # messages = [
    #     ("system", "You are a helpful translator. Translate the user sentence to French."),
    #     ("human", "I love programming."),
    # ]
    #
    # output = models[example_model_name][0].invoke(messages)
    # print(output)


if __name__ == '__main__':
    from extraction.methods.extending_model_outputs import add_all_extra_info_to_output

    main()
