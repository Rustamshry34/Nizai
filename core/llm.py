# agentforge/core/llm.py
from __future__ import annotations
import os
import json
import time
import asyncio
from typing import Any, Dict, Optional, AsyncIterator, List
from dataclasses import dataclass

# optional imports
try:
    import openai
except Exception:
    openai = None

try:
    import httpx
except Exception:
    httpx = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except Exception:
    # fallback
    def retry(*args, **kwargs):
        def _wrap(f): return f
        return _wrap
    stop_after_attempt = None
    wait_exponential = None
    retry_if_exception_type = None

# token counting
try:
    import tiktoken
except Exception:
    tiktoken = None


# --------------------------------------------------------
# LLMResponse
# --------------------------------------------------------
@dataclass
class LLMResponse:
    text: str
    raw: Any = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None


# --------------------------------------------------------
# BaseLLM interface
# --------------------------------------------------------
class BaseLLM:
    provider: str = "base"

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        raise NotImplementedError()

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        raise NotImplementedError()

    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        raise NotImplementedError()

    # optional function-calling aware method
    async def generate_with_functions(self, prompt: str, functions: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Adapter may implement this to return structured model outputs (e.g. OpenAI ChatCompletion dict).
        By default raise NotImplementedError to indicate unsupported.
        """
        raise NotImplementedError()


# --------------------------------------------------------
# LLM Stats
# --------------------------------------------------------
@dataclass
class LLMStats:
    calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0


# --------------------------------------------------------
# LLMClient: caching + retry + stats
# --------------------------------------------------------
class LLMClient:
    def __init__(self, model: str, adapter_kwargs: Optional[dict] = None, cache: Optional[dict] = None):
        self.model = model
        self.adapter_kwargs = adapter_kwargs or {}
        self.adapter = self._make_adapter(model, **self.adapter_kwargs)
        self.stats = LLMStats()
        self._cache = cache or {}

    def _make_adapter(self, model: str, **kwargs) -> BaseLLM:
        prefix = model.split(":", 1)[0] if ":" in model else "openai"

        if prefix == "openai":
            return OpenAIAdapter(model, **kwargs)
        if prefix in ("ollama", "local"):
            return OllamaAdapter(model, **kwargs)
        if prefix in ("mock", "test"):
            return MockAdapter(model)

        return MockAdapter(model)

    # ------------------------
    # Generate
    # ------------------------
    async def generate(self, prompt: str, use_cache: bool = True, cache_ttl: int = 300, **kwargs) -> LLMResponse:
        key = f"{self.model}|{prompt}|{json.dumps(kwargs, sort_keys=True)}"

        # cache hit
        if use_cache and key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["ts"] < cache_ttl:
                return entry["resp"]

        @retry(
            stop=stop_after_attempt(3) if stop_after_attempt else None,
            wait=wait_exponential(min=1, max=5) if wait_exponential else None,
            retry=retry_if_exception_type(Exception) if retry_if_exception_type else None,
        )
        async def _call():
            resp = await self.adapter.generate(prompt, **kwargs)
            self.stats.calls += 1
            if resp.tokens_used:
                self.stats.tokens += resp.tokens_used
            if resp.cost_usd:
                self.stats.cost_usd += resp.cost_usd
            return resp

        try:
            resp = await _call()
        except Exception:
            resp = await self.adapter.generate(prompt, **kwargs)

        if use_cache:
            self._cache[key] = {"ts": time.time(), "resp": resp}

        return resp

    # ------------------------
    # Function-calling aware generate (returns raw structured response)
    # ------------------------
    async def generate_with_functions(self, prompt: str, functions: List[Dict[str, Any]], use_cache: bool = True, cache_ttl: int = 300, **kwargs) -> Any:
        """
        Preferred path for function-calling: returns the adapter's raw structured response
        (e.g. OpenAI ChatCompletion response dict). If adapter doesn't implement it, raise.
        """
        key = f"{self.model}|FUNC|{prompt}|{json.dumps(functions, sort_keys=True)}|{json.dumps(kwargs, sort_keys=True)}"
        if use_cache and key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["ts"] < cache_ttl:
                return entry["resp"]

        @retry(
            stop=stop_after_attempt(3) if stop_after_attempt else None,
            wait=wait_exponential(min=1, max=5) if wait_exponential else None,
            retry=retry_if_exception_type(Exception) if retry_if_exception_type else None,
        )
        async def _call():
            # delegate to adapter
            resp = await self.adapter.generate_with_functions(prompt, functions=functions, **kwargs)
            # don't try to update tokens/cost here (adapter may include it)
            return resp

        try:
            resp = await _call()
        except NotImplementedError:
            # adapter doesn't support it
            raise
        except Exception:
            # final fallback: raise to caller
            resp = await self.adapter.generate_with_functions(prompt, functions=functions, **kwargs)

        if use_cache:
            self._cache[key] = {"ts": time.time(), "resp": resp}
        return resp

    # ------------------------
    # Stream
    # ------------------------
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        async for chunk in self.adapter.stream(prompt, **kwargs):
            yield chunk

    # ------------------------
    # Embedding
    # ------------------------
    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        return await self.adapter.embed(texts, **kwargs)


# --------------------------------------------------------
# OpenAI Adapter
# --------------------------------------------------------
class OpenAIAdapter(BaseLLM):
    provider = "openai"

    def __init__(self, model: str = "openai:gpt-3.5-turbo", temperature: float = 0.0):
        if openai is None:
            raise RuntimeError("openai package is not installed")

        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key

        self.model = model.split(":", 1)[1] if ":" in model else model
        self.temperature = temperature

    async def generate(self, prompt: str, max_tokens: int = 512, functions: Optional[List[Dict[str, Any]]] = None, **kwargs) -> LLMResponse:
        # if functions passed, route to function-calling method and wrap response appropriately
        if functions:
            raw = await self.generate_with_functions(prompt, functions=functions, **kwargs)
            # try to extract assistant text if available, else stringify raw
            try:
                choices = raw.get("choices", []) or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    if content:
                        text = content
                    else:
                        # no textual content (maybe function_call) -> stringify
                        text = json.dumps(raw)
                else:
                    text = json.dumps(raw)
            except Exception:
                text = json.dumps(raw)
            return LLMResponse(text=text, raw=raw)

        def _sync_call():
            messages = [{"role": "user", "content": prompt}]
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
            return resp

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, _sync_call)

        text = raw.choices[0].message.content
        usage = raw.get("usage", {})
        tokens = usage.get("total_tokens")
        cost = None  # optional

        return LLMResponse(text=text, raw=raw, tokens_used=tokens, cost_usd=cost)

    async def generate_with_functions(self, prompt: str, functions: List[Dict[str, Any]], max_tokens: int = 512, **kwargs) -> Any:
        """
        Use OpenAI function-calling feature. Returns the raw response dict from openai.ChatCompletion.create
        which includes choices[0].message.function_call if the model decided to call a function.
        """
        if openai is None:
            raise RuntimeError("openai package is not installed")

        def _sync_call():
            messages = [{"role": "user", "content": prompt}]
            # ChatCompletion supports functions param
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
                functions=functions,
                function_call="auto",  # let model choose
            )
            return resp

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, _sync_call)
        # return the raw dict-like response so planner can inspect function_call
        return raw

    async def stream(self, prompt: str, max_tokens: int = 512, **kwargs) -> AsyncIterator[str]:
        q = asyncio.Queue()

        def _thread():
            messages = [{"role": "user", "content": prompt}]
            for chunk in openai.ChatCompletion.create(
                model=self.model, messages=messages, max_tokens=max_tokens, stream=True
            ):
                for c in chunk.get("choices", []):
                    delta = c.get("delta", {})
                    txt = delta.get("content")
                    if txt:
                        asyncio.run_coroutine_threadsafe(q.put(txt), loop)
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

        loop = asyncio.get_event_loop()
        import threading
        t = threading.Thread(target=_thread, daemon=True)
        t.start()

        while True:
            piece = await q.get()
            if piece is None:
                break
            yield piece

    async def embed(self, texts: List[str], model: str = "text-embedding-3-small", **kwargs) -> List[List[float]]:
        def _sync():
            return openai.Embedding.create(model=model, input=texts)

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, _sync)
        return [d["embedding"] for d in raw["data"]]


# --------------------------------------------------------
# Ollama Adapter
# --------------------------------------------------------
class OllamaAdapter(BaseLLM):
    provider = "ollama"

    def __init__(self, model: str = "ollama:mistral", base_url: str = "http://localhost:11434", temperature: float = 0.0):
        if httpx is None:
            raise RuntimeError("httpx package required for Ollama")

        self.model = model.split(":", 1)[1] if ":" in model else model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    async def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> LLMResponse:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": self.temperature,
                },
            )
            r.raise_for_status()
            data = r.json()
            text = data.get("output") or data.get("text") or ""
            return LLMResponse(text=text, raw=data)

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": True},
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk.decode("utf-8", errors="ignore")

    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        raise NotImplementedError("OllamaAdapter.embed is not supported")


# --------------------------------------------------------
# Mock Adapter (for tests)
# --------------------------------------------------------
class MockAdapter(BaseLLM):
    provider = "mock"

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(text=f"[MOCK] {prompt}", raw={"mock": True})

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        for i in range(3):
            await asyncio.sleep(0.02)
            yield f"[MOCK_CHUNK_{i}] "
        yield "[END]"

    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        return [[float(len(t))] * 8 for t in texts]  # simple deterministic embedding

    async def generate_with_functions(self, prompt: str, functions: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Minimal mock behavior: return a fake structured response that *chooses* to call the first function
        with an example argument. This helps unit tests.
        """
        if not functions:
            return {"choices": [{"message": {"content": "[MOCK] no functions"}}]}
        func = functions[0]
        # craft a fake function_call structure like OpenAI returns
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": func.get("name"),
                            "arguments": json.dumps({list(func.get("parameters", {}).get("properties", {}).keys())[0]: "example"})
                        }
                    },
                    "finish_reason": "function_call"
                }
            ]
        }


# --------------------------------------------------------
# LLMAdapter (simple wrapper for LLMClient)
# --------------------------------------------------------
class LLMAdapter:
    """
    This is the unified interface used by the Agent:
    - generate() -> returns text
    - stream() -> async generator of tokens
    - embed() -> vector embeddings
    - generate_with_functions() -> returns structured raw response (adapter-dependent)
    """

    def __init__(self, model: str, **kwargs):
        self.client = LLMClient(model, adapter_kwargs=kwargs)
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        resp = await self.client.generate(prompt, **kwargs)
        # client.generate returns an LLMResponse
        return resp.text if isinstance(resp, LLMResponse) else str(resp)

    async def generate_with_functions(self, prompt: str, functions: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Exposes the client's generate_with_functions (returns structured raw response from adapter)
        """
        return await self.client.generate_with_functions(prompt, functions=functions, **kwargs)

    async def stream(self, prompt: str, **kwargs):
        async for token in self.client.stream(prompt, **kwargs):
            yield token

    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        return await self.client.embed(texts, **kwargs)


# --------------------------------------------------------
# Token counting helper
# --------------------------------------------------------
def count_tokens(model: str, text: str) -> Optional[int]:
    if tiktoken is None:
        return None
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
