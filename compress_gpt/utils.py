import sys

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from redis import StrictRedis as Redis
from rich import print


def has_redis():
    try:
        Redis().ping()
        return True
    except Exception:
        return False


def identity(x=None, *args):
    return (x,) + args if args else x


def wrap_prompt(name):
    upper = name.upper()
    return f"\n```start,name={upper}\n{{{name}}}\n```end,name={upper}"


def make_fast(model: ChatOpenAI) -> ChatOpenAI:
    if "turbo" in model.model_kwargs["model"]:
        return model

    return ChatOpenAI(
        temperature=model.temperature,
        verbose=model.verbose,
        streaming=model.streaming,
        callback_manager=model.callback_manager,
        model="gpt-3.5-turbo",
        request_timeout=model.request_timeout,
    )


class CompressCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        pass

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(
            f"\n[bold green]{prompts[0].splitlines()[1].strip()}[/bold green]\n",
            flush=True,
        )

    def on_llm_end(self, response, **kwargs):
        pass

    def on_llm_new_token(self, token, **kwargs):
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_llm_error(self, error, **kwargs):
        print(f"[bold red]{error}[/bold red]\n", flush=True)

    def on_chain_start(self, serialized, inputs, **kwargs):
        pass

    def on_chain_end(self, outputs, **kwargs):
        pass

    def on_chain_error(self, error, **kwargs):
        pass

    def on_tool_start(self, serialized, input_str, **kwargs):
        pass

    def on_agent_action(self, action, **kwargs):
        pass

    def on_tool_end(self, output, **kwargs):
        pass

    def on_tool_error(self, error, **kwargs):
        pass

    def on_text(self, text, end="", **kwargs):
        pass

    def on_agent_finish(self, finish, **kwargs):
        pass

    def flush_tracker(self, **kwargs):
        pass
