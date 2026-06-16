"""Quote-aware text tokenization utilities for PIERROT.

Quoted spans are treated as literal render strings: outside quotes uses normal
BPE tokenization, inside quotes is tokenized one Unicode character at a time.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from typing import Any, Sequence

import torch

_WORD_INTERNAL_QUOTE_RE = re.compile(r"[a-zA-Z]+'[a-zA-Z]+")


def split_quoted_spans(text: str) -> list[tuple[str, bool]]:
    """Split text into (span, is_quoted) while ignoring apostrophes inside words."""
    text = str(text or "")
    placeholders: list[tuple[str, str]] = []
    for i, word in enumerate(set(_WORD_INTERNAL_QUOTE_RE.findall(text))):
        key = f"pierrot_quote_placeholder_{i}_"
        text = text.replace(word, key)
        placeholders.append((word, key))

    quote_pairs = [("'", "'"), ('"', '"'), ('‘', '’'), ('“', '”')]
    quotes = ["'", '"', '‘', '’', '“', '”']
    for q1 in quotes:
        for q2 in quotes:
            if (q1, q2) not in quote_pairs:
                quote_pairs.append((q1, q2))
    pattern = '|'.join(
        re.escape(q1) + r'[^' + re.escape(q1 + q2) + r']*?' + re.escape(q2)
        for q1, q2 in quote_pairs
    )
    parts = re.split(f'({pattern})', text)
    out: list[tuple[str, bool]] = []
    for part in parts:
        if not part:
            continue
        for word, key in placeholders:
            part = part.replace(key, word)
        out.append((part, bool(re.fullmatch(pattern, part))))
    return out


def _encode_body_quote_char(tokenizer: Any, text: str) -> tuple[list[int], list[bool]]:
    ids: list[int] = []
    quoted_flags: list[bool] = []
    for span, is_quoted in split_quoted_spans(text):
        if is_quoted:
            for ch in span:
                ch_ids = tokenizer(ch, add_special_tokens=False).get('input_ids', [])
                ids.extend(ch_ids)
                quoted_flags.extend([True] * len(ch_ids))
        else:
            span_ids = tokenizer(span, add_special_tokens=False).get('input_ids', [])
            ids.extend(span_ids)
            quoted_flags.extend([False] * len(span_ids))
    return ids, quoted_flags


def _with_special_tokens(tokenizer: Any, body_ids: list[int]) -> list[int]:
    build = getattr(tokenizer, 'build_inputs_with_special_tokens', None)
    if callable(build):
        return list(build(body_ids))
    return list(body_ids)


def _special_overhead(tokenizer: Any) -> int:
    return max(0, len(_with_special_tokens(tokenizer, [])))


def _truncate_preserving_quoted(body_ids: list[int], quoted_flags: list[bool], max_body_len: int) -> list[int]:
    if len(body_ids) <= max_body_len:
        return body_ids
    if max_body_len <= 0:
        return []

    quoted_idx = [i for i, flag in enumerate(quoted_flags) if flag]
    if not quoted_idx:
        return body_ids[:max_body_len]
    if len(quoted_idx) >= max_body_len:
        keep = set(quoted_idx[:max_body_len])
        return [tok for i, tok in enumerate(body_ids) if i in keep]

    # Keep every quoted token, then spend the remaining budget on nearby context.
    # This protects appended clauses like: The visible text reads "OPEN".
    remaining = max_body_len - len(quoted_idx)
    nonquoted_idx = [i for i, flag in enumerate(quoted_flags) if not flag]

    def distance_to_quote(i: int) -> int:
        pos = bisect_left(quoted_idx, i)
        best = 10**9
        if pos < len(quoted_idx):
            best = min(best, abs(quoted_idx[pos] - i))
        if pos > 0:
            best = min(best, abs(quoted_idx[pos - 1] - i))
        return best

    ranked_context = sorted(nonquoted_idx, key=lambda i: (distance_to_quote(i), i))[:remaining]
    keep = set(quoted_idx) | set(ranked_context)
    return [tok for i, tok in enumerate(body_ids) if i in keep]


def encode_quote_char_ids(
    tokenizer: Any,
    text: str,
    max_length: int,
    preserve_quoted: bool = True,
    prefix: str | None = None,
) -> list[int]:
    body_ids, quoted_flags = _encode_body_quote_char(tokenizer, text)
    if prefix:
        # prefix (system prompt) 는 일반 BPE 로 앞에 붙이고 quoted=False 로 표시 → char-level/보존 대상 제외
        prefix_ids = tokenizer(prefix, add_special_tokens=False).get('input_ids', [])
        body_ids = list(prefix_ids) + body_ids
        quoted_flags = [False] * len(prefix_ids) + quoted_flags
    max_body_len = max(1, int(max_length) - _special_overhead(tokenizer))
    if preserve_quoted:
        body_ids = _truncate_preserving_quoted(body_ids, quoted_flags, max_body_len)
    else:
        body_ids = body_ids[:max_body_len]
    return _with_special_tokens(tokenizer, body_ids)[:max_length]


def tokenize_mixed_quote_char(
    tokenizer: Any,
    prompts: Sequence[str],
    max_length: int,
    quote_char_enabled: Sequence[bool] | None = None,
    preserve_quoted: bool = True,
    padding: bool | str = True,
    return_tensors: str = 'pt',
    prefix: str | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize a batch, applying quote-char encoding only where enabled.

    Args:
        prefix: None 이면 prompt 전체에 quote 판정/char-level 적용.   값이 있으면 (chi system prompt)
                각 prompt 의 prefix 부분은 일반 BPE 로 두고 그 뒤 caption 본문에만 char-level 적용
                → system prompt 안의 따옴표가 글리프 보존 대상에 섞이는 것 방지.

    옵션 A (2026-05-30) — 옛 코드 호환 보장:
        따옴표 없는 prompt 는 **batch 한 번에** tokenize → 옛 PIERROTPipeline 의 BPE batch 호출과 동일 path
        (huggingface tokenizer 의 batch vs per-prompt 의 미세 차이 제거 — diffusion sensitivity 회피).
        따옴표 있는 prompt 만 per-prompt char-level encode + 그 결과 batch padding merge.
    """
    if quote_char_enabled is None:
        quote_char_enabled = [True] * len(prompts)

    prefix = prefix or None

    def _body_of(p: Any) -> str:
        # prefix (system prompt) 떼고 caption 본문만 반환 — quote 판정은 본문에만.
        s = str(p) if p else ""
        if prefix and s.startswith(prefix):
            return s[len(prefix):]
        return s

    bodies = [_body_of(p) for p in prompts]                                          # quote 판정/처리 대상 = caption 본문

    # prompt 별로 char-level 활성 여부 판단 (caption 본문에 따옴표 있는 + enabled 만)
    needs_char = [
        bool(enabled and any(is_quoted for _, is_quoted in split_quoted_spans(body)))
        for body, enabled in zip(bodies, quote_char_enabled)
    ]

    # 모두 char-level 미적용 (= 따옴표 없는 prompt 만) → 옛 코드와 동일 batch tokenize path
    if not any(needs_char):
        return tokenizer(
            list(prompts),
            padding=padding,
            max_length=max_length if padding == 'max_length' else None,
            truncation=True,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )

    # 일부 char-level 적용 — per-prompt encode 후 batch padding merge
    encoded: list[list[int]] = []
    for prompt, body, char_on in zip(prompts, bodies, needs_char):
        has_prefix = bool(prefix and (str(prompt) if prompt else "").startswith(prefix))
        if char_on:
            ids = encode_quote_char_ids(
                tokenizer, body, max_length,
                preserve_quoted=preserve_quoted,
                prefix=prefix if has_prefix else None,
            )
        else:
            # non-char 샘플은 batch tokenize 경로처럼 원문 그대로 (strip X) → batch vs per-prompt 차이 최소화.
            raw = str(prompt) if (prompt is not None and str(prompt) != "") else " "
            ids = tokenizer(
                raw,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
            ).get('input_ids', [])
        encoded.append(ids or [getattr(tokenizer, 'eos_token_id', None) or 0])

    return tokenizer.pad(
        {'input_ids': encoded},
        padding=padding,
        max_length=max_length if padding == 'max_length' else None,
        return_attention_mask=True,
        return_tensors=return_tensors,
    )
