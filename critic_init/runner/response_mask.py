#!/usr/bin/env python
"""Build token-level masks that supervise only the assistant response.

The preferred input format contains ``prompt``, ``response`` and ``label``.  For
backward compatibility, Qwen/ChatML prompts without a separate ``response``
field are also supported by locating their final assistant header.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


ASSISTANT_MARKERS = (
    "<|im_start|>assistant\n",  # Qwen / ChatML
    "<|start_header_id|>assistant<|end_header_id|>\n\n",  # Llama 3
    "<｜Assistant｜>",  # DeepSeek
    "[/INST]",  # Llama 2 chat
)

ASSISTANT_END_MARKERS = (
    "<|im_end|>",
    "<|eot_id|>",
    "</s>",
)


def find_response_char_span(prompt: str, response: Optional[str] = None) -> tuple[int, int]:
    """Return the ``[start, end)`` character span of the assistant response."""
    if response is not None:
        if not response:
            raise ValueError("response must not be empty")
        start = prompt.rfind(response)
        if start < 0:
            raise ValueError("response is not an exact substring of prompt")
        return start, start + len(response)

    # Old critic-init JSONL files only contain the rendered chat prompt.  The
    # final assistant header is the response boundary for the templates below.
    marker_start = -1
    response_start = -1
    for marker in ASSISTANT_MARKERS:
        current = prompt.rfind(marker)
        if current > marker_start:
            marker_start = current
            response_start = current + len(marker)

    if response_start < 0:
        raise ValueError(
            "cannot infer the response span from prompt; add a top-level "
            "'response' field to the JSONL row"
        )

    response_end = len(prompt)
    for marker in ASSISTANT_END_MARKERS:
        current = prompt.find(marker, response_start)
        if current >= 0:
            response_end = min(response_end, current)
    return response_start, response_end


def mask_from_offsets(
    offsets: Sequence[Sequence[int]], response_start: int, response_end: int
) -> list[int]:
    """Convert tokenizer character offsets to a response-only token mask."""
    return [
        int(token_end > response_start and token_start < response_end)
        for token_start, token_end in offsets
    ]


def tokenize_with_response_mask(
    tokenizer,
    prompts: Sequence[str],
    responses: Optional[Sequence[Optional[str]]],
    *,
    max_length: int,
):
    """Tokenize a batch and add a full-sequence ``response_mask`` field."""
    if responses is not None and len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same batch size")

    tokenized = tokenizer(
        list(prompts),
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
    )
    offsets_batch = tokenized.pop("offset_mapping")

    response_masks = []
    for index, (prompt, offsets) in enumerate(zip(prompts, offsets_batch)):
        response = None if responses is None else responses[index]
        response_start, response_end = find_response_char_span(prompt, response)
        response_mask = mask_from_offsets(offsets, response_start, response_end)
        if not any(response_mask):
            raise ValueError(
                f"sample {index} has no response token after tokenization/truncation; "
                "increase --max_length or preserve more of the sequence suffix"
            )
        response_masks.append(response_mask)

    tokenized["response_mask"] = response_masks
    return tokenized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize critic-init JSONL and add a response-only token mask."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_length", type=int, default=16000)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("response-mask construction requires a fast tokenizer with offset mappings")

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_number, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            encoded = tokenize_with_response_mask(
                tokenizer,
                [row["prompt"]],
                [row.get("response")],
                max_length=args.max_length,
            )
            output_row = {
                **row,
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0],
                "response_mask": encoded["response_mask"][0],
            }
            fout.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    print(f"Wrote response masks to {output_path}")


if __name__ == "__main__":
    main()
