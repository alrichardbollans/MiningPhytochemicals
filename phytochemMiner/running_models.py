import os
import pickle

import langchain_core
import pydantic_core

from phytochemMiner import add_inchi_keys, read_file_and_chunk, standard_prompt, TaxaData, \
    deduplicate_and_standardise_output_taxa_lists, add_all_extra_info_to_output, get_txt_from_file


def run_phytochem_model(model, text_file: str, context_window: int, pkl_dump: str = None,
                        single_chunk: bool = True, rerun=True, rerun_inchi_resolution: bool = True) -> TaxaData:
    if not rerun and os.path.exists(pkl_dump):
        with open(pkl_dump, "rb") as file_:
            output = pickle.load(file_)
        if rerun_inchi_resolution:
            add_inchi_keys(output)
            with open(pkl_dump, "wb") as file_:
                pickle.dump(output, file_)
        return output

    if not single_chunk:
        raise NotImplementedError
    text_chunks = read_file_and_chunk(text_file, context_window)
    if single_chunk:
        # For most of analysis, will be testing on single chunks as this is how we've annotated them.
        # In this instance, the chunks should fit in the context window
        if not len(text_chunks) == 1:
            print(f'splitting chunks for {text_file}')
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
        if extraction is not None:
            if extraction.taxa is not None:
                output.extend(extraction.taxa)

    deduplicated_extractions = deduplicate_and_standardise_output_taxa_lists(output)
    add_all_extra_info_to_output(deduplicated_extractions)

    text = get_txt_from_file(text_file)
    deduplicated_extractions.text = text

    if pkl_dump:
        with open(pkl_dump, "wb") as file_:
            pickle.dump(deduplicated_extractions, file_)

    return deduplicated_extractions


def get_input_size_limit(total_context_window_k: int):
    # Output tokens so far is a tiny fraction, so allow 5% of context window for output
    out_units = total_context_window_k * 1000
    input_size = int(out_units * 0.95)
    return input_size


def get_phytochem_model(apikey: str = None, dotenv_path=None):
    """
    Gets an instance of the DeepSeek model configured with the specified parameters.

    The function provides an instance of the ChatDeepSeek model, either using the provided
    API key or loading it from an environment file. It specifies a default model type
    'deepseek-chat' with a predefined hyperparameter set for configuration.

    Parameters:
    apikey: str, optional
        The API key to authenticate with DeepSeek. If None, it retrieves the key from the
        environment file.
    dotenv_path: str, optional
        The file path to a `.env` file containing environment variables, including the API
        key under DEEPSEEK_API_KEY=. Defaults to None.

    Returns:
    tuple
        A tuple consisting of the configured DeepSeek model instance and the input size
        limit for the model.
    """

    # DeepSeek V3
    # Created 30/12/2024
    # Max tokens 128k
    # Input/Output: $0.28/0.42/1M tokens
    # https://api-docs.deepseek.com/quick_start/pricing/

    # DeepSeek-R1, specified via model="deepseek-reasoner", does not support tool calling or structured output.
    # Those features are supported by DeepSeek-V3 (specified via model="deepseek-chat").

    from langchain_deepseek import ChatDeepSeek
    hparams = {'temperature': 0}

    if apikey is None:
        # Get API keys
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=dotenv_path)

        model = ChatDeepSeek(
            model="deepseek-chat", **hparams)
    else:
        model = ChatDeepSeek(
            model="deepseek-chat", api_key=apikey, **hparams)
    return model, get_input_size_limit(128)
