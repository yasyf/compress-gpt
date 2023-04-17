from textwrap import dedent

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel

from compress_gpt.utils import wrap_prompt

from . import Prompt


class PromptComparison(BaseModel):
    discrepancies: list[str]
    equivalent: bool


class ComparePrompts(Prompt[PromptComparison]):
    @staticmethod
    def get_prompt() -> ChatPromptTemplate:
        system = SystemMessagePromptTemplate.from_template(
            dedent(
                """
            Inputs: restored prompt, analysis of diff from original prompt
            Task: Determine if restored is semantically equivalent to original

            Semantic equivalence means GPT-4 performs the same task with both prompts.
            This means GPT-4 needs the same understanding about the tools available, and the input & output formats.
            Significant differences in wording is ok, as long as equivalence is preserved.
            It is ok for the restored prompt to be more concise, as long as the output generated is similar.
            Differences in specificity that would generate a different result are discrepancies, and should be noted.
            Additional formatting instructions are provided. If these resolve a discrepancy, then do not include it.
            Not all diffs imply discrepancies. Discrepancies MUST be specific and present an obvious solution.

            Return your answer as a JSON object with the following schema:
            {{"discrepancies": [string], "equivalent": bool}}
        """
            )
        )
        human = HumanMessagePromptTemplate.from_template(
            wrap_prompt("restored")
            + "\n\n"
            + wrap_prompt("formatting")
            + "\n\n"
            + wrap_prompt("analysis")
        )
        return ChatPromptTemplate.from_messages([system, human])
