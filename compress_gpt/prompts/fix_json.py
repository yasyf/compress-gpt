from textwrap import dedent

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from compress_gpt.utils import wrap_prompt

from . import StrPrompt


class FixJSON(StrPrompt):
    @staticmethod
    def get_prompt() -> ChatPromptTemplate:
        task = SystemMessagePromptTemplate.from_template(
            dedent(
                """
            You will be provided with an invalid JSON string, and the error that was raised when parsing it.
            Return a valid JSON string by fixing any errors in the input. Be sure to fix any issues with backslash escaping.
            Do not include any explanation or commentary. Only return the fixed, valid JSON string.
            """
            )
        )
        human_1 = HumanMessagePromptTemplate.from_template(wrap_prompt("input"))
        human_2 = HumanMessagePromptTemplate.from_template(wrap_prompt("error"))
        return ChatPromptTemplate.from_messages([task, human_1, human_2])
