from functools import cached_property

from langchain import PromptTemplate

from compress_gpt.compress import Compressor


class CompressMixin:
    def _compress(self, prompt: str):
        return Compressor().compress(prompt)


class CompressPrompt(PromptTemplate, CompressMixin):
    def format(self, **kwargs) -> str:
        formatted = super().format(**kwargs)
        return self._compress(formatted)


class CompressTemplate(PromptTemplate, CompressMixin):
    @cached_property
    def template(self):
        return self._compress(super().template)
