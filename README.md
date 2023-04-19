# CompressGPT
## Self-extracting GPT prompts for ~70% token savings

Check out the accompanying blog post [here]().

### Demo

[![asciicast](https://asciinema.org/a/578285.svg)](https://asciinema.org/a/578285)

### Installation

```shell
$ pip install compress-gpt
```

### Usage

Simply change your existing imports of `langchain.PromptTemplate` to `compress_gpt.langchain.CompressTemplate` (to compress prompts before populating variables) or `compress_gpt.langchain.CompressPrompt` (to compress prompts after populating variables).

```diff
-from langchain import PromptTemplate
+from compress_gpt.langchain import CompressPrompt as PromptTemplate
```

For very simple prompts, use `CompressSimplePrompt` and `CompressSimpleTemplate` instead.

If compression ever fails or results in extra tokens, the original prompt will be used. Each compression result is aggressively cached, but the first run can take a hot sec.

### How CompressGPT Works

My [blog post]() helps explain the below image.

![CompressGPT Pipeline](assets/pipeline.svg)
