from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

compound_description = 'phytochemicals'

standard_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm tasked with extracting information about plants from scientific articles. "
            "Only extract relevant direct quotes from the text. Do not alter the extracted text by correcting spellings, expanding abbreviations or summarising the text. "
            "You should extract all scientific plant names mentioned in the text. "
            "You should include scientific authorities in the plant and fungal names if they appear in the text. "
            "Only extract scientific names. Do not extract common or vernacular names. "
            "For each of the plant names in the text, you should also extract mentions of any phytochemicals that are described as occurring in the plant. "
            "If a plant name appears without any associated phytochemicals, return null for the phytochemical attribute. "
        ),
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # MessagesPlaceholder("examples"),  # <-- EXAMPLES!
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        ("human", "{text}"),
    ]
)
