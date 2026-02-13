import functools
import hashlib
import json
import os
import sqlite3
import threading
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import httpx
import openai
from filelock import FileLock
from openai import AzureOpenAI, OpenAI
from packaging import version
from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

TextChatMessage = Dict[str, Any]


def _project_root_key_path() -> str:
    # .../src/structalignrag/llm/openai_compat.py -> project root
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "llm_key.txt"))


def maybe_set_llm_key_from_file(llm_base_url: str, default_base_url: str) -> None:
    """
    HippoRAG compatibility:
    - when using the default NVIDIA Integrate base_url and OPENAI_API_KEY is not set,
      set OPENAI_API_KEY from project-root llm_key.txt (first non-empty line).
    """
    if llm_base_url != default_base_url:
        return
    if os.getenv("OPENAI_API_KEY"):
        return
    key_path = _project_root_key_path()
    try:
        with open(key_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                os.environ["OPENAI_API_KEY"] = line
                return
    except FileNotFoundError:
        return


def _load_api_keys() -> List[str]:
    keys: List[str] = []
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        for raw_line in env_key.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                keys.extend(parts)
            else:
                keys.append(line)

    key_path = _project_root_key_path()
    try:
        with open(key_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "," in line:
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                    keys.extend(parts)
                else:
                    keys.append(line)
    except FileNotFoundError:
        pass

    seen = set()
    unique: List[str] = []
    for k in keys:
        if k and k not in seen:
            unique.append(k)
            seen.add(k)
    return unique


def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        messages = kwargs.get("messages") if "messages" in kwargs else (args[0] if args else None)
        if messages is None:
            raise ValueError("Missing 'messages' for caching.")

        gen_params = getattr(self, "generate_params", {})
        key_data = {
            "messages": messages,
            "model": kwargs.get("model", gen_params.get("model")),
            "seed": kwargs.get("seed", gen_params.get("seed")),
            "temperature": kwargs.get("temperature", gen_params.get("temperature")),
            "response_format": kwargs.get("response_format", gen_params.get("response_format")),
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        lock_file = self.cache_file + ".lock"
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
                """
            )
            conn.commit()
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                metadata["cache_hit"] = True
                return message, metadata

        message, metadata = func(self, *args, **kwargs)
        metadata = dict(metadata or {})
        metadata["cache_hit"] = False

        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
                """
            )
            c.execute(
                "INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                (key_hash, message, json.dumps(metadata)),
            )
            conn.commit()
            conn.close()

        return message, metadata

    return wrapper


def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 2)
        decorated = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))(func)
        return decorated(self, *args, **kwargs)

    return wrapper


class CacheOpenAICompat:
    """
    OpenAI-compatible chat completion client with:
    - key rotation (multiple keys in llm_key.txt or OPENAI_API_KEY env)
    - sqlite cache (per outputs/<dataset>/llm_cache)
    """

    def __init__(
        self,
        cache_dir: str,
        llm_name: str,
        llm_base_url: str,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        seed: Optional[int] = None,
        max_retries: int = 2,
        high_throughput: bool = True,
        azure_endpoint: Optional[str] = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.llm_name = llm_name
        self.llm_base_url = llm_base_url
        self.max_retries = max_retries

        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, f"llm_cache_{self.llm_name.replace('/', '_')}.sqlite")

        # Generation defaults (can be overridden per-call)
        self.generate_params: Dict[str, Any] = {
            "model": self.llm_name,
            "temperature": temperature,
            "seed": seed,
            # OpenAI protocol variants: some servers accept max_completion_tokens, others max_tokens.
            "max_completion_tokens": max_new_tokens,
        }

        limits = httpx.Limits(max_keepalive_connections=100, max_connections=100) if high_throughput else None
        timeout = httpx.Timeout(timeout_s, connect=timeout_s) if high_throughput else httpx.Timeout(timeout_s)
        http_client = httpx.Client(limits=limits, timeout=timeout) if high_throughput else None

        self._client_lock = threading.Lock()
        self._client_index = 0
        self._client_pool: List[OpenAI] = []

        if azure_endpoint:
            # AzureOpenAI api_version should be included in azure_endpoint query string (?api-version=...)
            api_version = azure_endpoint.split("api-version=")[1]
            self.openai_client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                max_retries=self.max_retries,
            )
            return

        api_keys = _load_api_keys()
        if not api_keys:
            self.openai_client = OpenAI(base_url=self.llm_base_url, http_client=http_client, max_retries=self.max_retries)
        else:
            self._client_pool = [
                OpenAI(api_key=k, base_url=self.llm_base_url, http_client=http_client, max_retries=self.max_retries)
                for k in api_keys
            ]
            self.openai_client = self._client_pool[0]

    def _get_openai_client(self) -> OpenAI:
        if not self._client_pool:
            return self.openai_client
        with self._client_lock:
            client = self._client_pool[self._client_index % len(self._client_pool)]
            self._client_index += 1
            return client

    def num_keys(self) -> int:
        """
        Return how many API keys are available for concurrent throughput.
        When no explicit keys are configured, treat as a single-key client.
        """
        return len(self._client_pool) if self._client_pool else 1

    @cache_response
    @dynamic_retry_decorator
    def infer(self, messages: List[TextChatMessage], **kwargs) -> Tuple[str, Dict[str, Any]]:
        params = deepcopy(self.generate_params)
        params.update(kwargs or {})
        params["messages"] = messages

        # Compatibility: openai>=1.45.0 changed param naming; vllm/openai-compat may not support max_completion_tokens.
        if "max_completion_tokens" in params:
            if ("gpt" not in str(params.get("model", ""))) or version.parse(openai.__version__) < version.parse("1.45.0"):
                params["max_tokens"] = params.pop("max_completion_tokens")

        client = self._get_openai_client()
        resp = client.chat.completions.create(**params)
        content = resp.choices[0].message.content
        if not isinstance(content, str):
            content = str(content)

        meta = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "finish_reason": resp.choices[0].finish_reason,
        }
        return content, meta
