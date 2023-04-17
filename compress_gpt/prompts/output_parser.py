import asyncio
import re
from typing import Generic, Optional, Type, TypeVar, Union, cast, get_args

import dirtyjson
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, ValidationError, parse_obj_as, validator

from compress_gpt.utils import make_fast

TModel = TypeVar("TModel", bound=Type[BaseModel])
TModelList = TypeVar("TModelList", bound=list[Type[BaseModel]])
TM = Union[TModel, TModelList]
M = TypeVar("M", bound=TM)


class OutputParser(PydanticOutputParser, Generic[M]):
    format: Optional[M] = None
    model: ChatOpenAI

    @validator("format", always=True)
    def set_format(cls, _, values: dict) -> Type[BaseModel]:
        return values["pydantic_object"]

    @validator("pydantic_object", always=True)
    def set_pydantic_object(cls, obj: M) -> Type[BaseModel]:
        return get_args(obj)[0] if isinstance(obj, list) else obj

    def _preprocess(self, text: str) -> str:
        text = re.sub(
            re.compile(r"([^\\])\\([^\\nt\"])"), lambda m: f"{m[1]}\\\\{m[2]}", text
        )
        if isinstance(self.format, list) and text.startswith("{"):
            text = f"[{text}]"
        return text

    async def _fix(self, text: str, error: str) -> str:
        from .fix_json import FixJSON

        return await FixJSON.run(model=make_fast(self.model), input=text, error=error)

    async def aparse(
        self, text: str, attempts: int = 3
    ) -> Union[BaseModel, list[BaseModel]]:
        for _ in range(attempts):
            try:
                text = self._preprocess(text)
                parsed = dirtyjson.loads(text, search_for_first_object=True)
                return parse_obj_as(cast(M, self.format), parsed)
            except (dirtyjson.Error, ValidationError) as e:
                text = await self._fix(text, str(e))

        return super().parse(text)

    def parse(self, text: str) -> Union[BaseModel, list[BaseModel]]:
        return asyncio.run(self.aparse(text))
