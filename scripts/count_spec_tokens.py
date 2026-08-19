"""Token-count check for prompts/specs/*.md (PROMPT_SPEC.md #3.2: ~5k tokens
+-10% per language, counted on the *injectable* content only -- everything
after the first '---' line, since the block above it is provenance metadata
stripped by spec_loader.py before the spec is put in a prompt).

Uses tiktoken's cl100k_base encoding as a stable, model-agnostic proxy for
"tokens" (exact counts differ per model tokenizer, but this is the standard
approximation used for prompt budgeting).
"""
import sys
import tiktoken

TARGET = 5000
TOLERANCE = 0.10

enc = tiktoken.get_encoding("cl100k_base")


def injectable_content(path):
    text = open(path).read()
    marker = "\n---\n"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError(f"{path}: no '---' provenance/content separator found")
    return text[idx + len(marker):]


def main():
    paths = sys.argv[1:] or [
        "prompts/specs/cuda.md",
        "prompts/specs/ptx.md",
        "prompts/specs/triton.md",
        "prompts/specs/tilelang.md",
    ]
    lo, hi = TARGET * (1 - TOLERANCE), TARGET * (1 + TOLERANCE)
    rc = 0
    for p in paths:
        content = injectable_content(p)
        n = len(enc.encode(content))
        ok = lo <= n <= hi
        print(f"{p}: {n} tokens {'OK' if ok else 'OUT OF BUDGET (' + f'{lo:.0f}-{hi:.0f}' + ')'}")
        rc |= 0 if ok else 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
