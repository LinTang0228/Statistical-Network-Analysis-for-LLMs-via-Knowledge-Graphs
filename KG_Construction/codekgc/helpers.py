"""LLM call helpers for CodeKGC (OpenAI and Anthropic backbones)."""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


_OPENAI_CLIENT = None
def _openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI()   # reads OPENAI_API_KEY
    return _OPENAI_CLIENT


def call_openai(prompt: str, model: str, temperature: float, max_tokens: int,
                max_retries: int = 3) -> str:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = _openai_client().chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"call_openai failed after {max_retries} retries: {last_err}")


_ANTHROPIC_CLIENT = None
def _anthropic_client():
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        import anthropic
        _ANTHROPIC_CLIENT = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
    return _ANTHROPIC_CLIENT


def call_claude(prompt: str, model: str, temperature: float, max_tokens: int,
                max_retries: int = 3) -> str:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = _anthropic_client().messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"call_claude failed after {max_retries} retries: {last_err}")


def parallel_call(call_fn, prompts, *, model, temperature, max_tokens, concurrency=4):
    """Run `call_fn` over `prompts` in parallel; return list of (completion, error)
    in input order."""
    results = {}

    def _do(idx_prompt):
        i, prompt = idx_prompt
        try:
            return i, call_fn(prompt, model=model, temperature=temperature,
                              max_tokens=max_tokens), None
        except Exception as e:
            return i, "", f"{type(e).__name__}: {e}"

    n_done = 0
    t0 = time.time()
    print(f"  calling LLM on {len(prompts)} prompts (concurrency={concurrency})...")
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_do, (i, p)) for i, p in enumerate(prompts)]
        for fut in as_completed(futures):
            i, completion, err = fut.result()
            results[i] = (completion, err)
            n_done += 1
            if n_done % 25 == 0 or n_done == len(prompts):
                rate = n_done / max(1e-9, time.time() - t0)
                print(f"    {n_done}/{len(prompts)}   ({rate:.1f} items/s)")
    print(f"  done in {time.time() - t0:.1f}s")
    return [results[i] for i in range(len(prompts))]
