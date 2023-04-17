from textwrap import dedent

from langchain.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from compress_gpt.prompts.compress_chunks import CompressChunks
from compress_gpt.utils import wrap_prompt

from . import StrPrompt


class IdentifyFormat(StrPrompt):
    @staticmethod
    def get_prompt() -> ChatPromptTemplate:
        CompressChunks.get_prompt().messages[0]
        task = SystemMessagePromptTemplate.from_template(
            dedent(
                """
                Task: Filter the input provided by the user.

                Proccess the input below one line at a time.
                Each line is an instruction for a large language model.
                For each line, decide whether to keep or discard it.

                Rules:
                Discard lines not needed to infer the output format.
                Discard lines that are about the task to be performed, unless they mention how to format output.
                Keep lines that describe the structure of the output.
                Keep any lines needed to infer response structure.
                Keep any explicit examples of response structure.
                Keep any lines that show how to invoke tools.
                Keep any lines that describe a JSON or other schema.

                Returns:
                Output each kept line as you process it.
            """
            )
        )
        ex_human = HumanMessagePromptTemplate.from_template(
            dedent(
                """
                Here is an example:
                ```start,name=INPUT
                Your job is to take a list of addresses, and extract the components of each.
                The components are the street name, the city, and the state.

                Context:
                    Date: 2021-01-01
                    Time: 12:00:00
                    User: John Doe

                ALWAYS return your output in the following format:
                [{{"street": "123 Main St", "city": "New York", "state": "NY"}}]

                Do not include duplicates. Do not include any streets in CA.

                Your output should be a list of valid JSON objects.
                ```end,name=INPUT
            """
            )
        )
        ex_ai = AIMessagePromptTemplate.from_template(
            dedent(
                """
                ALWAYS return your output in the following format:
                [{{"street": "123 Main St", "city": "New York", "state": "NY"}}]

                Your output should be a list of valid JSON objects.
            """
            )
        )
        human = HumanMessagePromptTemplate.from_template(
            "This is the input to process:\n" + wrap_prompt("input")
        )
        return ChatPromptTemplate.from_messages([task, ex_human, ex_ai, human])
