import os
import sys
from datetime import timedelta
from functools import partial
from pathlib import Path

import langchain
import nest_asyncio
from aiocache import Cache, cached
from aiocache.serializers import PickleSerializer
from langchain.cache import RedisCache, SQLiteCache
from redis import Redis

from compress_gpt.utils import has_redis, identity

nest_asyncio.apply()

CACHE_DIR = Path(os.getenv("XDG_CACHE_HOME", "~/.cache")).expanduser() / "compress-gpt"

if "pytest" in sys.modules:
    langchain.llm_cache = None
    cache = lambda: identity  # noqa: E731
elif has_redis():
    langchain.llm_cache = RedisCache(redis_=Redis())
    cache = partial(
        cached,
        ttl=timedelta(days=7),
        cache=Cache.REDIS,
        serializer=PickleSerializer(),
        noself=True,
    )
else:
    langchain.llm_cache = SQLiteCache(
        database_path=str(CACHE_DIR / "langchain.db"),
    )
    cache = partial(
        cached,
        cache=Cache.MEMORY,
        serializer=PickleSerializer(),
        noself=True,
    )

from .compress import Compressor as Compressor
