from abc import ABC, abstractmethod
from typing import Generic, Optional, Type, cast, get_args

from langchain import LLMChain
from langchain.callbacks import AimCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain.schema import BaseLanguageModel

from .output_parser import M, OutputParser


class Prompt(ABC, Generic[M]):
    @staticmethod
    @abstractmethod
    def get_prompt() -> ChatPromptTemplate:
        ...

    @classmethod
    def get_format(cls) -> Type[M]:
        return get_args(cls.__orig_bases__[0])[0]

    @classmethod
    def get_chain(cls, model: Optional[BaseLanguageModel]):
        model = model or ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        prompt = cls.get_prompt()
        prompt.output_parser = OutputParser[M](
            pydantic_object=cls.get_format(), model=model
        )
        return LLMChain(llm=model, prompt=prompt)

    @classmethod
    async def run(cls, model: Optional[BaseLanguageModel] = None, **kwargs):
        chain = cls.get_chain(model=model)
        try:
            return cast(M, await chain.apredict_and_parse(**kwargs))
        finally:
            if aim := next(
                (
                    h
                    for h in chain.llm.callback_manager.handlers
                    if isinstance(h, AimCallbackHandler)
                ),
                None,
            ):
                aim.flush_tracker(langchain_asset=chain, reset=True, finish=False)


class StrPrompt(Prompt[str]):
    @classmethod
    def get_chain(cls, *args, **kwargs):
        chain = super().get_chain(*args, **kwargs)
        chain.prompt.output_parser = None
        return chain


from .compress_chunks import CompressChunks as CompressChunks
