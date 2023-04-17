from textwrap import dedent

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from compress_gpt.utils import wrap_prompt

from . import StrPrompt


class DiffPrompts(StrPrompt):
    @staticmethod
    def get_prompt() -> ChatPromptTemplate:
        system = SystemMessagePromptTemplate.from_template(
            dedent(
                """
            There are two sets of instructions being considered.
            Your task is to diff the two sets of instructions to understand their functional differences.
            Differences in clarity, conciseness, or wording are not relevant, UNLESS they imply a functional difference.

            These are the areas to diff:
            - The intent of the task to perform
            - Factual information provided
            - Instructions to follow
            - The specifc tools available, and how exactly to use them
            - The input and output, focusing on the schema and format
            - Conditions and constraints

            Generate a diff of the two prompts, by considering each of the above areas.
            Be very specific in your diffing. You must diff every aspect of the two prompts.
        """
            )
        )
        human = HumanMessagePromptTemplate.from_template(
            wrap_prompt("original") + "\n\n" + wrap_prompt("restored")
        )
        return ChatPromptTemplate.from_messages([system, human])
