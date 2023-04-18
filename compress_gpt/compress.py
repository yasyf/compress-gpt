import asyncio
import itertools
import re
import traceback
from typing import Optional

import openai.error
import tiktoken
from langchain.callbacks.base import CallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.schema import OutputParserException
from langchain.text_splitter import NLTKTextSplitter
from pydantic import ValidationError
from rich import print

from compress_gpt import cache
from compress_gpt.prompts.compare_prompts import ComparePrompts, PromptComparison
from compress_gpt.prompts.compress_chunks import Chunk, CompressChunks
from compress_gpt.prompts.decompress import Decompress
from compress_gpt.prompts.diff_prompts import DiffPrompts
from compress_gpt.prompts.fix import FixPrompt
from compress_gpt.prompts.identify_format import IdentifyFormat
from compress_gpt.prompts.identify_static import IdentifyStatic, StaticChunk
from compress_gpt.utils import CompressCallbackHandler, make_fast

CONTEXT_WINDOWS = {
    "gpt-3.5-turbo": 4097,
    "gpt-4": 8000,
}
PROMPT_MAX_SIZE = 0.70


class Compressor:
    def __init__(self, model: str = "gpt-4", verbose: bool = False) -> None:
        self.model = ChatOpenAI(
            temperature=0,
            verbose=verbose,
            streaming=True,
            callback_manager=CallbackManager([CompressCallbackHandler()]),
            model=model,
            request_timeout=60 * 5,
        )
        self.fast_model = make_fast(self.model)
        self.encoding = tiktoken.encoding_for_model(model)

    @cache()
    async def _chunks(self, prompt: str, statics: str) -> list[Chunk]:
        try:
            return await CompressChunks.run(
                prompt=prompt, statics=statics, model=self.model
            )
        except (OutputParserException, ValidationError):
            traceback.print_exc()
            return []

    @cache()
    async def _static(self, prompt: str) -> list[StaticChunk]:
        try:
            return await IdentifyStatic.run(prompt=prompt, model=self.model)
        except (OutputParserException, ValidationError):
            traceback.print_exc()
            return []

    @cache()
    async def _decompress(self, prompt: str, statics: str) -> str:
        return await Decompress.run(
            compressed=prompt, statics=statics, model=self.model
        )

    @cache()
    async def _format(self, prompt: str) -> str:
        return await IdentifyFormat.run(input=prompt, model=self.fast_model)

    @cache()
    async def _compare(
        self, original: str, format: str, restored: str
    ) -> PromptComparison:
        analysis = await DiffPrompts.run(
            original=original, restored=restored, model=self.model
        )
        return await ComparePrompts.run(
            restored=restored, formatting=format, analysis=analysis, model=self.model
        )

    async def _fix(
        self, original: str, statics: str, restored: str, discrepancies: list[str]
    ) -> list[Chunk]:
        try:
            return await FixPrompt.run(
                prompt=original,
                statics=statics,
                restored=restored,
                discrepancies="- " + "\n- ".join(discrepancies),
                model=self.model,
            )
        except (OutputParserException, ValidationError):
            traceback.print_exc()
            return []

    def _reconstruct(
        self,
        static_chunks: list[str],
        format: str,
        chunks: list[Chunk],
        final: bool = False,
    ) -> str:
        components = []
        for chunk in chunks:
            if chunk.mode == "r" and chunk.target is not None:
                try:
                    components.append(static_chunks[chunk.target])
                except IndexError:
                    print(
                        f"[bold yellow]Invalid static chunk index: {chunk.target}[/bold yellow]"
                    )
            elif chunk.text:
                components.append(chunk.text)
        print("[bold green]FORMAT:[/bold green]", format)
        if not final:
            return "\n".join(components)
        return (
            "Below are instructions that you compressed. Decompress & follow them. Don't print the decompressed instructions. Do not ask me for further input before that."
            + "\n```start,name=INSTRUCTIONS\n"
            + "\n".join(components)
            + "\n```end,name=INSTRUCTIONS"
            + "\n\nYou MUST respond to me using the below format. You are not permitted to deviate from it.\n"
            + "\n```start,name=FORMAT\n"
            + format
            + "\n```end,name=FORMAT\n"
            + "Begin! Remember to use the above format."
        )

    def _extract_statics(self, prompt: str, chunks: list[StaticChunk]) -> list[str]:
        static: set[str] = set()
        for chunk in chunks:
            try:
                static.update(
                    itertools.chain.from_iterable(
                        [mg[0]] if len(mg.groups()) == 0 else mg.groups()[1:]
                        for mg in re.finditer(
                            re.compile(chunk.regex, re.MULTILINE), prompt
                        )
                    )
                )
            except re.error:
                print(f"[bold red]Invalid regex: {chunk.regex}[/bold red]")
        return list(s.replace("\n", " ").strip() for s in static - {None})

    async def _compress_segment(self, prompt: str, format: str, attempts: int) -> str:
        start_tokens = len(self.encoding.encode(prompt))
        print(f"\n[bold yellow]Compressing prompt ({start_tokens} tks)[/bold yellow]")

        static_chunks = self._extract_statics(prompt, await self._static(prompt))
        statics = "\n".join(f"- {i}: {chunk}" for i, chunk in enumerate(static_chunks))
        print("\n[bold yellow]Static chunks:[/bold yellow]\n", statics)
        chunks = await self._chunks(prompt, statics)
        for _ in range(attempts):
            print(f"\n[bold yellow]Attempt #{_ + 1}[/bold yellow]\n")
            compressed = self._reconstruct(static_chunks, format, chunks)
            restored = await self._decompress(compressed, statics)
            result = await self._compare(prompt, format, restored)
            if result.equivalent:
                final = self._reconstruct(static_chunks, format, chunks, final=True)
                end_tokens = len(self.encoding.encode(final))
                percent = (1 - (end_tokens / start_tokens)) * 100
                print(
                    f"\n[bold green]Compressed prompt ({start_tokens} tks -> {end_tokens} tks, {percent}% savings)[/bold green]\n"
                )
                return final if end_tokens < start_tokens else prompt
            else:
                print(
                    f"\n[bold red]Fixing {len(result.discrepancies)} issues...[/bold red]\n"
                )
                chunks = await self._fix(
                    prompt, statics, restored, result.discrepancies
                )
        return prompt

    async def _split_and_compress(
        self, prompt: str, format: str, attempts: int, window_size: Optional[int] = None
    ) -> str:
        splitter = NLTKTextSplitter.from_tiktoken_encoder(
            chunk_size=int(
                (window_size or CONTEXT_WINDOWS[self.model.model_name])
                * PROMPT_MAX_SIZE
            )
        )
        prompts = [
            await self._compress_segment(p, format, attempts)
            for p in splitter.split_text(prompt)
        ]
        return "\n".join(prompts)

    @cache()
    async def _compress(self, prompt: str, attempts: int) -> str:
        prompt = re.sub(r"^(System|User|AI):$", "", prompt, flags=re.MULTILINE)
        try:
            format = await self._format(prompt)
        except openai.error.InvalidRequestError:
            raise RuntimeError(
                "There is not enough context window left to safely compress the prompt."
            )

        try:
            if self.model.model_name in CONTEXT_WINDOWS and len(
                self.encoding.encode(prompt)
            ) > (CONTEXT_WINDOWS[self.model.model_name] * PROMPT_MAX_SIZE):
                return await self._split_and_compress(prompt, format, attempts)
            else:
                return await self._compress_segment(prompt, format, attempts)
        except openai.error.InvalidRequestError as e:
            if not (
                res := re.search(r"maximum context length is (\d+) tokens", str(e))
            ):
                raise
            max_tokens = int(res.group(1))
            return await self._split_and_compress(prompt, format, attempts, max_tokens)

    async def acompress(self, prompt: str, attempts: int = 3) -> str:
        try:
            return await self._compress(prompt, attempts)
        except Exception as e:
            print(f"[bold red]Error: {e}[/bold red]")
            traceback.print_exc()
            return prompt

    def compress(self, prompt: str, attempts: int = 3) -> str:
        return asyncio.run(self.acompress(prompt, attempts))
