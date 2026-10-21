"""DeepSeek client shared by the LLM agents (05-08).

DeepSeek's API is OpenAI-compatible, so this is a few lines of httpx rather
than an SDK. Needs DEEPSEEK_API_KEY in the environment. Override the model
with WEATHERAGENTS_MODEL (default deepseek-flash; deepseek-v4-pro is the
bigger one).
"""

import json
import os
import time

import httpx
from pydantic import BaseModel, ValidationError

DEFAULT_MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com"


def model_name() -> str:
    return os.environ.get("WEATHERAGENTS_MODEL", DEFAULT_MODEL)


class LLM:
    def __init__(self, api_key: str, *, base_url: str = BASE_URL, model: str | None = None):
        self.model = model or model_name()
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=300,
        )

    def _complete(self, payload: dict) -> dict:
        for attempt in range(4):
            resp = self._http.post("/chat/completions", json={"model": self.model, **payload})
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]
        raise RuntimeError("unreachable")

    def chat(self, *, system: str, user: str, max_tokens: int = 3000) -> str:
        msg = self._complete(
            {
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens,
            }
        )
        return msg.get("content") or ""

    def structured(self, *, system: str, user: str, schema: type[BaseModel], max_tokens: int = 8000, attempts: int = 3) -> BaseModel:
        """One forced tool call whose arguments are validated against `schema`.

        The model occasionally emits malformed or off-schema JSON; retry rather than fail the event.
        """
        payload = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},  # forced tool_choice is rejected in thinking mode
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "submit",
                        "description": "Submit your final answer.",
                        "parameters": schema.model_json_schema(),
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "submit"}},
        }
        last: Exception | None = None
        for _ in range(attempts):
            msg = self._complete(payload)
            calls = msg.get("tool_calls") or []
            if not calls:
                last = RuntimeError("model returned no tool call")
                continue
            try:
                return schema.model_validate(json.loads(calls[0]["function"]["arguments"]))
            except (json.JSONDecodeError, ValidationError) as exc:
                last = exc
        raise RuntimeError(f"no valid structured answer after {attempts} attempts: {type(last).__name__}: {last}")


def make_client() -> LLM:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set; the LLM agents (05-08) need it")
    return LLM(key)
