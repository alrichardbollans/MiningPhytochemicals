# Following https://python.langchain.com/docs/how_to/extraction_examples/

import uuid
from typing import List, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from pydantic import BaseModel

from extraction.methods.structured_output_schema import TaxaData, Taxon


class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted


def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    openai_tool_calls = []
    for tool_call in example["tool_calls"]:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    # The name of the function right now corresponds
                    # to the name of the pydantic model
                    # This is implicit in the API right now,
                    # and will be improved over time.
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages


_examples = [
    (
        "Panax ginseng C.A.Mey has many benefits; it has been used for tiredness and enhancement of physical performance.",
        TaxaData(taxa=[Taxon(scientific_name='Panax ginseng C.A.Mey', medical_conditions=['tiredness'],
                             medicinal_effects=['enhancement of physical performance'])]),
    ),
    (
        "The opium poppy (Papaver somniferum L.) is grown in many places around the world. It is commonly used to treat hypertension. It is also used for pain relief as well as vasodilation. Valeriana officinalis is a perennial flowering plant native to Eurasia, which is often used for treating insomnia.",
        TaxaData(taxa=[
            Taxon(scientific_name='Papaver somniferum L.', medical_conditions=['hypertension'], medicinal_effects=['vasodilation', 'pain relief']),
            Taxon(scientific_name='Valeriana officinalis', medical_conditions=['insomnia'], medicinal_effects=None)]),
    ),
    (
        "The perennial flowering plant Ricinus communis is a very effective purgative. ",
        TaxaData(taxa=[Taxon(scientific_name='Ricinus communis', medical_conditions=None, medicinal_effects=['purgative'])]),
    ),
    (
        "Quinine is isolated from the bark of the cinchona tree (Cinchona officinalis L.), and this plant is used to treat malaria. The cinchona tree has been known to be used traditionally for fever, for its antifebrile effects.",
        TaxaData(taxa=[Taxon(scientific_name='Cinchona officinalis L.', medical_conditions=['malaria', 'fever'],
                             medicinal_effects=['antifebrile'])]),
    ),
    (
        "During the tanning process, the animal skin is soaked in a tannin extraction for a period of time ranging from just a few hours to several months.",
        TaxaData(taxa=[Taxon(scientific_name=None, medical_conditions=None, medicinal_effects=None)]),
    )
]

example_messages = []

for text, tool_call in _examples:
    example_messages.extend(
        tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
    )
