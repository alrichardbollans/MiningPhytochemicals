from langchain_text_splitters import TokenTextSplitter

from phytochemMiner import remove_double_spaces_and_break_characters


def get_txt_from_file(txt_file: str):
    import os

    with open(os.path.join(txt_file), "r", encoding="utf8") as f:
        text = f.read()

    out = remove_double_spaces_and_break_characters(text)

    return out


def read_file_and_chunk(txt_file: str, context_size: int) -> list:
    # chunking specific to models: https://python.langchain.com/v0.1/docs/use_cases/extraction/how_to/handle_long_text/
    text = get_txt_from_file(txt_file)
    text_splitter = TokenTextSplitter(
        # Controls the size of each chunk
        chunk_size=context_size,
        # Controls overlap between chunks
        chunk_overlap=500,
    )

    texts = text_splitter.split_text(text)

    return texts


def split_text_chunks(text_chunks, overlap=500):
    """
    Split each text chunk in a list while retaining a specified amount of overlap.

    This is useful when text has already been chunked but needs splitting again

    Parameters:
    text_chunks (list): A list of text chunks to be split.
    overlap (int): The number of characters to overlap between the split chunks.

    Returns:
    list: A list of the split text chunks.
    """
    split_chunks = []
    for chunk in text_chunks:
        chunk_length = len(chunk)
        split_point = int(chunk_length / 2)
        first_half = chunk[:split_point + int(overlap / 2)]
        second_half = chunk[split_point - int(overlap / 2):]
        split_chunks.extend([first_half, second_half])
    return split_chunks
