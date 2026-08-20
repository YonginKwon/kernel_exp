"""Parses prompts/PROMPT_SPEC.md into the templates scripts/generate.py uses.

This is the ONLY place that reads PROMPT_SPEC.md's structure. Per the project
rule ("PROMPT_SPEC.md가 프롬프트의 유일한 진실"), no template text is
duplicated/hand-copied anywhere else in the codebase -- generate.py always
goes through build_prompt()/build_repair_prompt() below, so a PROMPT_SPEC.md
edit takes effect everywhere automatically.

Parsing approach: PROMPT_SPEC.md's templates live in fenced code blocks under
known headings. One block (the common template, section 1) contains a NESTED
```python fence around {REFERENCE_CODE} -- so blocks are extracted with a
fence-depth counter (any ``` line toggles depth; the block is everything
between the first 0->1 transition and the matching last 1->0 transition),
not a naive "first ``` ... ``` pair" scan.
"""
import re
from pathlib import Path

SPEC_PATH = Path(__file__).parent / "PROMPT_SPEC.md"
SPECS_DIR = Path(__file__).parent / "specs"

LANGUAGE_DISPLAY = {
    "cuda": "CUDA",
    "triton": "Triton",
    "ptx": "PTX",
    "tilelang": "TileLang",
}
SPEC_FILE_FOR_LANGUAGE = {
    "cuda": "cuda.md",
    "triton": "triton.md",
    "ptx": "ptx.md",
    "tilelang": "tilelang.md",
}

_FENCE_RE = re.compile(r"^```")


def _extract_fenced_block(text: str) -> str:
    """First fenced block in `text`, handling one level of nested fencing
    (see module docstring). Returns the block's inner text, fence lines
    stripped from the outside but preserved if nested inside."""
    return _extract_with_toggle(text.splitlines())


def _extract_with_toggle(lines: list[str]) -> str:
    """Unambiguous extraction: track fence lines as a stack where every
    ``` toggles between 'expecting open' and 'expecting close' at each
    depth. Depth sequence in PROMPT_SPEC.md is always: open(0->1),
    [open(1->2), close(2->1)]*, close(1->0)."""
    depth = 0
    start = None
    content_lines = []
    for line in lines:
        is_fence = _FENCE_RE.match(line) is not None
        if is_fence:
            if depth == 0:
                depth = 1
                start = True
                continue
            elif depth == 1:
                # Could be a nested open or the final close. Peek: a nested
                # open always has a language tag (```python) or is followed
                # by more non-fence content before another close; the final
                # close always ends the block. We disambiguate by scanning
                # ahead in the caller instead -- here, treat bare ``` at
                # depth 1 as CLOSE, and ```<lang> at depth 1 as nested OPEN.
                if line.strip() == "```":
                    depth = 0
                    break
                else:
                    depth = 2
                    content_lines.append(line)
                    continue
            elif depth == 2:
                depth = 1
                content_lines.append(line)
                continue
        else:
            if depth >= 1:
                content_lines.append(line)
    if start is None:
        raise ValueError("no fenced block found")
    return "\n".join(content_lines)


def _section(text: str, heading_pattern: str) -> str:
    """Text of the section starting at a heading matching `heading_pattern`
    (regex, matched at line start) up to the next heading of the same or
    higher level (## or ###, whichever the matched heading itself uses)."""
    lines = text.splitlines()
    level = None
    start = None
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+", line)
        if m and re.match(heading_pattern, line):
            level = len(m.group(1))
            start = i + 1
            break
    if start is None:
        raise ValueError(f"heading not found: {heading_pattern}")
    end = len(lines)
    for i in range(start, len(lines)):
        m = re.match(r"^(#{1,6})\s+", lines[i])
        if m and len(m.group(1)) <= level:
            end = i
            break
    return "\n".join(lines[start:end])


class PromptSpec:
    def __init__(self, path: Path = SPEC_PATH):
        self.raw = path.read_text()
        self.common_template = _extract_fenced_block(_section(self.raw, r"^## 1\."))
        lang_section = _section(self.raw, r"^## 2\.")
        self.language_blocks = {}
        for key, display in LANGUAGE_DISPLAY.items():
            block_text = _section(lang_section, rf"^### {re.escape(display)}\s*$")
            self.language_blocks[key] = _extract_fenced_block(block_text).strip("\n")
        cond_section = _section(self.raw, r"^## 3\.")
        doc_inject_section = _section(cond_section, r"^### 3\.2\b")
        self.doc_injection_template = _extract_fenced_block(doc_inject_section).strip("\n")
        repair_section = _section(cond_section, r"^### 3\.3\b")
        self.repair_template = _extract_fenced_block(repair_section).strip("\n")

        # §3.4 multi-turn protocol (2026-08-20, supersedes §3.3's single
        # repair-only template) -- each of the 5 templates below lives under
        # its own named heading in PROMPT_SPEC.md specifically so it can be
        # parsed by name here, not by position in a bullet list.
        multiturn_section = _section(cond_section, r"^### 3\.4\b")
        repair_phase_section = _section(multiturn_section, r"^#### 수리 국면 피드백")
        self.repair_compile_template = _extract_fenced_block(
            _section(repair_phase_section, r"^##### 컴파일 실패")).strip("\n")
        self.repair_runtime_template = _extract_fenced_block(
            _section(repair_phase_section, r"^##### 런타임 실패")).strip("\n")
        self.repair_correctness_template = _extract_fenced_block(
            _section(repair_phase_section, r"^##### 정확성 실패")).strip("\n")
        self.repair_parse_failure_template = _extract_fenced_block(
            _section(repair_phase_section, r"^##### 파싱 실패")).strip("\n")
        self.optimization_template = _extract_fenced_block(
            _section(multiturn_section, r"^#### 최적화 국면 피드백")).strip("\n")

        gen_params_section = _section(self.raw, r"^## 4\.")
        self.generation_params = _parse_generation_params(gen_params_section)

    def build_prompt(self, language: str, reference_code: str, doc_spec_text: str | None = None) -> str:
        if language not in LANGUAGE_DISPLAY:
            raise ValueError(f"unknown language {language!r}, expected one of {list(LANGUAGE_DISPLAY)}")
        display = LANGUAGE_DISPLAY[language]
        lang_block = self.language_blocks[language]
        if doc_spec_text is not None:
            injected = self.doc_injection_template.replace("{LANGUAGE_SPEC_5K}", doc_spec_text)
            lang_block = lang_block + "\n\n" + injected
        prompt = self.common_template
        prompt = prompt.replace("{LANGUAGE}", display)
        prompt = prompt.replace("{LANGUAGE_BLOCK}", lang_block)
        prompt = prompt.replace("{REFERENCE_CODE}", reference_code)
        return prompt.strip("\n") + "\n"

    def build_repair_prompt(self, compiler_error: str) -> str:
        return self.repair_template.replace("{COMPILER_ERROR_VERBATIM}", compiler_error).strip("\n") + "\n"

    # --- §3.4 multi-turn feedback builders -----------------------------
    # Each returns ONLY the feedback text -- the caller (scripts/multiturn.py)
    # concatenates {original turn-1 task prompt} + {previous turn's code} +
    # {this feedback}, per §3.4's "히스토리 무상태" rule (no chat history,
    # a single freshly-assembled prompt every turn).

    def build_repair_compile_feedback(self, compiler_error: str) -> str:
        return self.repair_compile_template.replace(
            "{COMPILER_ERROR_VERBATIM}", compiler_error).strip("\n") + "\n"

    def build_repair_runtime_feedback(self, runtime_error: str) -> str:
        return self.repair_runtime_template.replace(
            "{RUNTIME_ERROR_VERBATIM}", runtime_error).strip("\n") + "\n"

    def build_repair_correctness_feedback(self, max_abs_error, mismatch_fraction) -> str:
        return (self.repair_correctness_template
                .replace("{MAX_ABS_ERROR}", str(max_abs_error))
                .replace("{MISMATCH_FRACTION}", str(mismatch_fraction))
                .strip("\n") + "\n")

    def build_repair_parse_failure_feedback(self) -> str:
        return self.repair_parse_failure_template.strip("\n") + "\n"

    def build_optimization_feedback(self, kernel_ms, baseline_ms, speedup) -> str:
        return (self.optimization_template
                .replace("{X}", f"{kernel_ms:.4g}")
                .replace("{Y}", f"{baseline_ms:.4g}")
                .replace("{Z}", f"{speedup:.3g}")
                .strip("\n") + "\n")


def _parse_generation_params(text: str) -> dict:
    params = {}
    m = re.search(r"temperature:\s*([\d.]+)", text)
    if m:
        params["temperature"] = float(m.group(1))
    m = re.search(r"샘플 수:\s*(\d+)", text)
    if m:
        params["num_samples"] = int(m.group(1))
    m = re.search(r"max output tokens:\s*(\d+)", text)
    if m:
        params["max_output_tokens"] = int(m.group(1))
    return params


def load_language_spec(language: str) -> str:
    """The injectable text of prompts/specs/<language>.md -- everything after
    the first '---' line (the block above it is provenance metadata, not
    part of what gets put in a prompt). Used for the doc-injection ablation
    condition (PROMPT_SPEC.md §3.2)."""
    path = SPECS_DIR / SPEC_FILE_FOR_LANGUAGE[language]
    text = path.read_text()
    marker = "\n---\n"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError(f"{path}: no '---' provenance/content separator found")
    return text[idx + len(marker):]


def get_spec() -> PromptSpec:
    return PromptSpec()
