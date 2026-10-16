"""llm.py -- Groq (panelists) and Gemini (auditor) API calls, TPM pacing, retries.
Other modules call these as llm.call_groq / llm.call_gemini so tests can replace them."""
import random
import re
import time
from collections import deque

import requests

import config as cfg


class DailyLimitError(RuntimeError):
    """Raised when the provider reports a per-day quota is exhausted (no point retrying)."""


class AuditorUnavailableError(RuntimeError):
    """Gemini stayed unavailable for the whole retry budget. Progress is checkpointed; rerun to resume."""


class TPMPacer:
    """Sliding 60s window over actual token usage (Groq TPM counts prompt + completion)."""

    def __init__(self, limit, safety=0.85):
        self.budget = limit * safety
        self.window = deque()  # (timestamp, tokens)

    def _prune(self, now):
        while self.window and now - self.window[0][0] >= 60:
            self.window.popleft()

    def wait(self, est_tokens):
        while True:
            now = time.time()
            self._prune(now)
            used = sum(t for _, t in self.window)
            if not self.window or used + est_tokens <= self.budget:
                return
            sleep_for = 60 - (now - self.window[0][0]) + 0.5
            print(f"  [pacing: ~{used} tokens in last 60s, waiting {sleep_for:.0f}s]")
            time.sleep(max(sleep_for, 0.5))

    def record(self, tokens):
        self.window.append((time.time(), tokens))


def call_groq(system_prompt, user_prompt, pacer, seed, temperature,
              max_tokens=cfg.MAX_TOKENS_PER_TURN, retries=6):
    if not cfg.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    headers = {"Authorization": f"Bearer {cfg.GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": cfg.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,  # best-effort determinism; logged either way
    }
    est_tokens = int((len(system_prompt) + len(user_prompt)) / 3.5) + 250

    for attempt in range(retries):
        pacer.wait(est_tokens)
        try:
            resp = requests.post(cfg.GROQ_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                err_msg = ""
                try:
                    err_msg = resp.json().get("error", {}).get("message", "")
                except Exception:
                    pass
                if re.search(r"per day|\bTPD\b|\bRPD\b", err_msg, re.IGNORECASE):
                    raise DailyLimitError(
                        "Groq DAILY limit reached -- stop for today and rerun this run later. "
                        f"Provider message: {err_msg}"
                    )
                wait = None
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        pass
                if wait is None:
                    match = re.search(r"try again in ([\d\.]+)s", err_msg)
                    if match:
                        wait = float(match.group(1)) + 1.0
                if wait is None:
                    wait = 8 * (attempt + 1)
                print(f"  [rate limited (429), backing off {wait:.1f}s (attempt {attempt + 1}/{retries})]")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"  [Groq API Error {resp.status_code}]: {resp.text}")
                resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage") or {}
            pacer.record(usage.get("total_tokens", est_tokens))
            return {
                "text": data["choices"][0]["message"].get("content") or "",
                "usage": usage,
                "model": data.get("model"),
                "fingerprint": data.get("system_fingerprint"),
            }
        except DailyLimitError:
            raise
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Groq API call failed: {e}")
            time.sleep(3)

    raise RuntimeError("Groq API call failed after retries.")


def call_gemini(system_prompt, user_prompt):
    if not cfg.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    headers = {"x-goog-api-key": cfg.GEMINI_API_KEY, "Content-Type": "application/json"}
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": cfg.AUDITOR_TEMPERATURE,
            "maxOutputTokens": cfg.AUDITOR_MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": cfg.AUDITOR_THINKING_BUDGET},
        },
    }
    started, attempt = time.time(), 0
    while True:
        attempt += 1
        status, body, resp = None, "", None
        try:
            resp = requests.post(cfg.GEMINI_URL, headers=headers, json=payload, timeout=120)
            status, body = resp.status_code, resp.text
        except requests.exceptions.RequestException as e:
            body = str(e)  # timeout / connection error: treated as transient

        if status == 200:
            data = resp.json()
            cands = data.get("candidates") or []
            parts = ((cands[0].get("content") or {}).get("parts") or []) if cands else []
            return {
                "text": "".join(p.get("text", "") for p in parts),
                "usage": data.get("usageMetadata"),
                "model": data.get("modelVersion"),
                "finish_reason": cands[0].get("finishReason") if cands else None,
            }
        if status == 429 and re.search(r"per day|PerDay", body[:300], re.IGNORECASE):
            raise DailyLimitError(f"Gemini DAILY limit reached. Provider message: {body[:300]}")
        if status is not None and status not in (429, 500, 502, 503, 504):
            print(f"  [Gemini API Error {status}]: {body}")
            raise RuntimeError(f"Gemini API error {status}: {body[:300]}")  # not transient, don't retry

        wait = min(cfg.GEMINI_BACKOFF_CAP_S, cfg.GEMINI_BACKOFF_BASE_S * 2 ** (attempt - 1)) + random.uniform(0, 5)
        elapsed = time.time() - started
        print(f"  [Gemini {status or 'network error'}: {body[:200]!r} | attempt {attempt}, {elapsed:.0f}s elapsed]")
        if elapsed + wait > cfg.GEMINI_RETRY_BUDGET_S:
            raise AuditorUnavailableError(
                f"Gemini unavailable after {attempt} attempts / {elapsed:.0f}s (last: {status} {body[:200]})")
        print(f"  [retrying in {wait:.0f}s]")
        time.sleep(wait)