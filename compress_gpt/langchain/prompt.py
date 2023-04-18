from functools import cached_property

from langchain import PromptTemplate
from pydantic import BaseModel

from compress_gpt.compress import Compressor


class CompressMixin(BaseModel):
    compressor_kwargs: dict = {}

    def _compress(self, prompt: str):
        return Compressor(**self.compressor_kwargs).compress(prompt)

    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (cached_property,)


class CompressPrompt(CompressMixin, PromptTemplate):
    def format(self, **kwargs) -> str:
        formatted = super().format(**kwargs)
        return self._compress(formatted)


class CompressTemplate(CompressMixin, PromptTemplate):
    @cached_property
    def template(self):
        return self._compress(super().template)


class CompressSimplePrompt(CompressPrompt):
    compressor_kwargs = {"complex": False}


class CompressSimpleTemplate(CompressTemplate):
    compressor_kwargs = {"complex": False}
