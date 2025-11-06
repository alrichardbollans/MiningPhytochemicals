from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

compound_description = 'phytochemicals'

standard_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm tasked with extracting information about occurrences of phytochemicals from scientific articles. "
            "For each of the plant names in the text, you should extract mentions of any phytochemicals that are described as occurring in the plant. "
            "Only extract information where plant scientific names are provided. Do not extract common or vernacular names. "
            "You should include scientific authorities in the plant names if they appear in the text. "
            "You should extract information about specific compounds only, not generic compound classes. "
            "In general you should not modify the extracted text, however where two similar compounds are labelled to"
            " distinguish variants you should extract the variants separately, e.g. for abibalsamins A and B extract abibalsamins A and abibalsamins B. "
            "Often compounds will be numbered in the text, but do not extract these numbers. "
            "Do not alter the extracted text by correcting spellings, expanding abbreviations or summarising the text. "
            "If a text provides no information on phytochemicals, return null. "
        ),
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # MessagesPlaceholder("examples"),  # <-- EXAMPLES!
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        ("human", "{text}"),
    ]
)

if __name__ == '__main__':
    print(standard_prompt.messages[0].prompt.template)
