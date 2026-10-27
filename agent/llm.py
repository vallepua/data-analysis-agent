# agent/llm.py
"""Single place where we talk to Groq. Handles rate limits (429) politely."""
import os
import re
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

_client = None
_last_call = 0.0


def _get_key():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:  # Streamlit Cloud secrets
            import streamlit as st
            key = st.secrets["GROQ_API_KEY"]
        except Exception:
            key = None
    return key


def _get_client():
    global _client
    if _client is None:
        key = _get_key()
        if not key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
        _client = Groq(api_key=key)
    return _client


def _retry_seconds(message):
    """Read 'try again in 1.6s' / '7m12.5s' from a Groq error message."""
    m = re.search(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)s", message)
    if not m:
        return 20.0
    minutes = float(m.group(1) or 0)
    return minutes * 60 + float(m.group(2))


def call_llm(prompt, max_tokens=1500, system=None, retries=4):
    """Return the model's text. Raises RuntimeError if it cannot be reached."""
    global _last_call
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    use_effort = "gpt-oss" in MODEL
    last_error = ""

    for attempt in range(retries):
        gap = 2.5 - (time.time() - _last_call)   # stay under requests/minute
        if gap > 0:
            time.sleep(gap)
        try:
            kwargs = dict(model=MODEL, messages=messages,
                          max_tokens=max_tokens, temperature=0.2)
            if use_effort:
                kwargs["reasoning_effort"] = "low"   # leaves tokens for the answer
            resp = _get_client().chat.completions.create(**kwargs)
            _last_call = time.time()
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa
            _last_call = time.time()
            last_error = str(e)
            low = last_error.lower()
            if use_effort and "reasoning" in low:
                use_effort = False            # model does not accept the option
                continue
            if "429" in last_error or "rate_limit" in low:
                wait = _retry_seconds(last_error)
                if wait > 90:
                    raise RuntimeError(
                        "Groq usage limit reached. Please wait a few minutes and try again."
                    )
                time.sleep(wait + 1)
                continue
            if attempt < retries - 1 and ("503" in last_error or "timeout" in low):
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"AI call failed: {last_error[:200]}")
    raise RuntimeError(f"AI call failed after retries: {last_error[:200]}")
