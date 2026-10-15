#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build critic JSONL in raw, answer, or solution context mode.

``raw`` uses question + model response. ``answer`` appends the ground-truth
answer to the question. ``solution`` appends both the ground-truth answer and
reference solutions looked up by the stripped exact question string. When a
question has multiple solutions, ``--multi_solution_mode first`` keeps only
the first one, while ``--multi_solution_mode all`` puts every solution into
the same critic prompt.

Balancing first downsamples all-correct question groups by a configurable
factor (default 4), then upsamples the global minority label. Mixed and
all-wrong question groups are never discarded.

Examples (run from the repository root)::

    # Raw critic: question + model response
    python critic_init/scripts/convert_infer_to_critic_jsonl.py \
        --mode raw \
        --input_file /path/to/inference.jsonl \
        --output_file /path/to/raw_critic_train.jsonl \
        --model_name_or_path /path/to/base_model \
        --overwrite

    # Answer critic: question + ground-truth answer + model response
    python critic_init/scripts/convert_infer_to_critic_jsonl.py \
        --mode answer \
        --input_file /path/to/inference.jsonl \
        --output_file /path/to/answer_critic_train.jsonl \
        --model_name_or_path /path/to/base_model \
        --overwrite

    # Solution critic: question + answer + reference solution + model response
    python critic_init/scripts/convert_infer_to_critic_jsonl.py \
        --mode solution \
        --multi_solution_mode first \
        --input_file /path/to/inference.jsonl \
        --reference_file dataset/deepmath/deepmath_train.json \
        --output_file /path/to/critic_train.jsonl \
        --model_name_or_path /path/to/base_model \
        --overwrite

    # Solution critic with every matched solution in the same prompt
    python critic_init/scripts/convert_infer_to_critic_jsonl.py \
        --mode solution \
        --multi_solution_mode all \
        --input_file /path/to/inference.jsonl \
        --reference_file dataset/deepmath/deepmath_train.json \
        --output_file /path/to/critic_train_all_solutions.jsonl \
        --model_name_or_path /path/to/base_model \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer

try:
    import ijson
except ImportError:
    ijson = None


MODES = ("raw", "answer", "solution")
MULTI_SOLUTION_MODES = ("first", "all")


@dataclass(frozen=True)
class Candidate:
    question: str
    response: str
    label: float
    answer: Optional[str] = None
    upsample_eligible: bool = True


def parse_csv_fields(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def detect_input_format(path: Path, input_format: str) -> str:
    if input_format in {"json", "jsonl"}:
        return input_format
    if path.suffix.lower() == ".jsonl":
        return "jsonl"
    if path.suffix.lower() == ".json":
        return "json"
    with path.open("r", encoding="utf-8") as input_file:
        return "json" if input_file.read(4096).lstrip().startswith("[") else "jsonl"


def first_non_whitespace(path: Path) -> str:
    with path.open("r", encoding="utf-8") as input_file:
        while True:
            char = input_file.read(1)
            if not char or not char.isspace():
                return char


def iter_records(path: Path, input_format: str) -> Iterator[Dict[str, Any]]:
    """Stream JSONL and top-level JSON arrays; support small wrapped JSON too."""
    fmt = detect_input_format(path, input_format)
    if fmt == "jsonl":
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {line_number} is not an object")
                yield row
        return

    if first_non_whitespace(path) == "[" and ijson is not None:
        with path.open("rb", buffering=8 * 1024 * 1024) as input_file:
            for row_index, row in enumerate(ijson.items(input_file, "item", use_float=True)):
                if not isinstance(row, dict):
                    raise ValueError(f"JSON record {row_index} is not an object")
                yield row
        return

    with path.open("r", encoding="utf-8") as input_file:
        obj = json.load(input_file)
    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict) and isinstance(obj.get("data"), list):
        rows = obj["data"]
    elif isinstance(obj, dict):
        rows = [obj]
    else:
        raise ValueError("JSON must be an object, list[object], or {'data': list[object]}")
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"JSON record {row_index} is not an object")
        yield row


def get_nested(row: Dict[str, Any], field: str) -> Any:
    value: Any = row
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def is_usable(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def pick_first(row: Dict[str, Any], fields: Iterable[str]) -> Tuple[Optional[Any], Optional[str]]:
    for field in fields:
        value = get_nested(row, field)
        if is_usable(value):
            return value, field
    return None, None


def question_from_messages(value: Any) -> Optional[str]:
    if not isinstance(value, list):
        return None
    for message in reversed(value):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "user")).strip().lower() != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    for message in reversed(value):
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def extract_question(row: Dict[str, Any], fields: Sequence[str]) -> Optional[str]:
    for field in fields:
        value = get_nested(row, field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        question = question_from_messages(value)
        if question:
            return question
    return None


def str_to_bool(text: str) -> Optional[bool]:
    normalized = text.strip().lower()
    if normalized in {"1", "true", "yes", "y", "t"}:
        return True
    if normalized in {"0", "false", "no", "n", "f"}:
        return False
    return None


def to_binary_label(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if float(value) > 0.0 else 0.0
    if isinstance(value, str):
        parsed_bool = str_to_bool(value)
        if parsed_bool is not None:
            return 1.0 if parsed_bool else 0.0
        try:
            return 1.0 if float(value.strip()) > 0.0 else 0.0
        except ValueError as exc:
            raise ValueError(f"cannot parse label string {value!r}") from exc
    raise ValueError(f"unsupported label type: {type(value)}")


def normalize_solutions(value: Any) -> List[str]:
    raw_items = value if isinstance(value, list) else [value]
    solutions: List[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        if raw is None:
            continue
        text = raw.strip() if isinstance(raw, str) else str(raw).strip()
        if text and text not in seen:
            seen.add(text)
            solutions.append(text)
    return solutions


def collect_candidates(
    path: Path,
    input_format: str,
    mode: str,
    question_fields: Sequence[str],
    answer_fields: Sequence[str],
    response_fields: Sequence[str],
    label_fields: Sequence[str],
    drop_empty_response: bool,
) -> Tuple[List[Candidate], Counter[str]]:
    candidates: List[Candidate] = []
    stats: Counter[str] = Counter()
    for row in iter_records(path, input_format):
        stats["total_rows"] += 1
        question = extract_question(row, question_fields)
        response_raw, _ = pick_first(row, response_fields)
        label_raw, _ = pick_first(row, label_fields)
        answer_raw: Any = None
        if mode in {"answer", "solution"}:
            answer_raw, _ = pick_first(row, answer_fields)
        if question is None or response_raw is None or label_raw is None:
            stats["skipped_missing_field"] += 1
            continue
        if mode in {"answer", "solution"} and answer_raw is None:
            stats["skipped_missing_field"] += 1
            continue
        response = str(response_raw).strip()
        if drop_empty_response and not response:
            stats["skipped_empty_response"] += 1
            continue
        candidates.append(
            Candidate(
                question=question,
                response=response,
                label=to_binary_label(label_raw),
                answer=str(answer_raw).strip() if answer_raw is not None else None,
            )
        )
    stats["usable_rows"] = len(candidates)
    return candidates, stats


def downsample_all_correct_questions(
    candidates: Sequence[Candidate], factor: int, seed: int
) -> Tuple[List[Candidate], Counter[str]]:
    """Keep ceil(N/factor) responses for groups whose labels are all positive."""
    if factor < 1:
        raise ValueError("--all_correct_downsample_factor must be >= 1")
    groups: Dict[str, List[int]] = defaultdict(list)
    for candidate_index, candidate in enumerate(candidates):
        groups[candidate.question].append(candidate_index)

    rng = random.Random(seed)
    keep_indices: set[int] = set()
    all_correct_kept_indices: set[int] = set()
    stats: Counter[str] = Counter()
    for member_indices in groups.values():
        if all(candidates[index].label == 1.0 for index in member_indices):
            stats["all_correct_questions"] += 1
            stats["all_correct_rows_before"] += len(member_indices)
            keep_count = max(1, (len(member_indices) + factor - 1) // factor)
            selected = (
                member_indices
                if keep_count == len(member_indices)
                else rng.sample(member_indices, keep_count)
            )
            keep_indices.update(selected)
            all_correct_kept_indices.update(selected)
            stats["all_correct_rows_kept"] += len(selected)
        else:
            keep_indices.update(member_indices)

    stats["all_correct_rows_dropped"] = (
        stats["all_correct_rows_before"] - stats["all_correct_rows_kept"]
    )
    kept: List[Candidate] = []
    for candidate_index, candidate in enumerate(candidates):
        if candidate_index not in keep_indices:
            continue
        # Do not undo the all-correct downsampling when selecting upsample copies.
        kept.append(
            replace(candidate, upsample_eligible=False)
            if candidate_index in all_correct_kept_indices
            else candidate
        )
    stats["rows_after_all_correct_downsample"] = len(kept)
    return kept, stats


def load_solution_index(
    paths: Sequence[Path],
    input_format: str,
    question_fields: Sequence[str],
    solution_fields: Sequence[str],
    needed_questions: set[str],
) -> Dict[str, List[str]]:
    """Build exact-question -> ordered, de-duplicated solution list."""
    index: Dict[str, List[str]] = {}
    seen: Dict[str, set[str]] = {}
    stats: Counter[str] = Counter()
    for path in paths:
        for row in iter_records(path, input_format):
            stats["reference_rows"] += 1
            question = extract_question(row, question_fields)
            if not question:
                stats["skipped_missing_reference_question"] += 1
                continue
            if question not in needed_questions:
                continue
            raw_solution, _ = pick_first(row, solution_fields)
            solutions = normalize_solutions(raw_solution)
            if not solutions:
                stats["skipped_missing_reference_solution"] += 1
                continue
            if question in index:
                stats["duplicate_question_rows"] += 1
            else:
                index[question] = []
                seen[question] = set()
            stats["matched_reference_rows"] += 1
            for solution in solutions:
                if solution not in seen[question]:
                    seen[question].add(solution)
                    index[question].append(solution)
    stats["matched_questions"] = len(index)
    stats["indexed_solutions"] = sum(len(items) for items in index.values())
    print("[solution_index] " + " ".join(f"{key}={stats[key]}" for key in sorted(stats)))
    return index


def select_solutions_for_prompt(
    solutions: Sequence[str], multi_solution_mode: str
) -> Tuple[str, int]:
    """Return one solution block and the number of included solutions."""
    if not solutions:
        raise ValueError("cannot build a solution prompt from an empty solution list")
    if multi_solution_mode == "first":
        return solutions[0], 1
    if multi_solution_mode == "all":
        if len(solutions) == 1:
            return solutions[0], 1
        combined = "\n\n".join(
            f"Solution {solution_index}:\n{solution}"
            for solution_index, solution in enumerate(solutions, start=1)
        )
        return combined, len(solutions)
    raise ValueError(f"unsupported multi-solution mode: {multi_solution_mode}")


def output_class_counts(candidates: Sequence[Candidate]) -> Counter[float]:
    counts: Counter[float] = Counter()
    for candidate in candidates:
        # Both multi-solution modes produce exactly one prompt per candidate.
        counts[candidate.label] += 1
    return counts


def plan_upsampling(
    candidates: Sequence[Candidate],
    seed: int,
) -> Tuple[Counter[Tuple[int, int]], Counter[float]]:
    """Sample minority output rows with replacement to reach global 1:1."""
    class_counts = output_class_counts(candidates)
    positive = class_counts[1.0]
    negative = class_counts[0.0]
    if positive == negative:
        return Counter(), class_counts
    if positive == 0 or negative == 0:
        raise ValueError(
            "cannot balance because one label is absent: "
            f"positive={positive}, negative={negative}"
        )
    minority_label = 1.0 if positive < negative else 0.0
    target = max(positive, negative)

    eligible_candidate_indices: List[int] = []
    for candidate_index, candidate in enumerate(candidates):
        if candidate.label != minority_label or not candidate.upsample_eligible:
            continue
        eligible_candidate_indices.append(candidate_index)
    if not eligible_candidate_indices:
        raise ValueError(
            "the minority class has no upsample-eligible samples after excluding "
            "all-correct question groups"
        )

    rng = random.Random(seed)
    repeats: Counter[Tuple[int, int]] = Counter()
    additions = target - min(positive, negative)
    for _ in range(additions):
        candidate_index = rng.choice(eligible_candidate_indices)
        repeats[(candidate_index, 0)] += 1
    return repeats, class_counts


def build_prompt_from_messages(
    tokenizer: AutoTokenizer,
    mode: str,
    question: str,
    response: str,
    system_prompt: Optional[str],
    answer: Optional[str] = None,
    solution: Optional[str] = None,
    solution_count: int = 1,
) -> str:
    if mode == "raw":
        user_content = question
    elif mode == "answer":
        if not answer:
            raise ValueError("answer mode requires a non-empty ground-truth answer")
        user_content = f"{question} The ground truth answer is {answer}."
    elif mode == "solution":
        if not answer:
            raise ValueError("solution mode requires a non-empty ground-truth answer")
        if not solution:
            raise ValueError("solution mode requires a non-empty reference solution")
        solution_heading = (
            "A reference solution:" if solution_count == 1 else "Reference solutions:"
        )
        user_content = (
            f"{question} The ground truth answer is {answer}."
            f"\n\n{solution_heading}\n{solution}"
        )
    else:
        raise ValueError(f"unsupported mode: {mode}")

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": response})
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if not isinstance(prompt, str):
        raise TypeError(f"apply_chat_template returned non-string: {type(prompt)}")
    return prompt


def write_output(
    output_path: Path,
    candidates: Sequence[Candidate],
    mode: str,
    solution_map: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    system_prompt: Optional[str],
    extra_repetitions: Counter[Tuple[int, int]],
    multi_solution_mode: str,
) -> Counter[float]:
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    written: Counter[float] = Counter()
    try:
        with temporary_path.open("w", encoding="utf-8", buffering=8 * 1024 * 1024) as output_file:
            for candidate_index, candidate in enumerate(candidates):
                solution: Optional[str] = None
                solution_count = 0
                if mode == "solution":
                    solution, solution_count = select_solutions_for_prompt(
                        solution_map[candidate.question], multi_solution_mode
                    )
                prompt = build_prompt_from_messages(
                    tokenizer=tokenizer,
                    mode=mode,
                    question=candidate.question,
                    answer=candidate.answer,
                    solution=solution,
                    solution_count=solution_count,
                    response=candidate.response,
                    system_prompt=system_prompt,
                )
                output = {
                    "prompt": prompt,
                    "response": candidate.response,
                    "label": candidate.label,
                }
                line = json.dumps(output, ensure_ascii=False) + "\n"
                repetitions = 1 + extra_repetitions[(candidate_index, 0)]
                for _ in range(repetitions):
                    output_file.write(line)
                written[candidate.label] += repetitions
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return written


def build_parser(default_mode: str = "raw") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default=default_mode)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--input_format", choices=("auto", "json", "jsonl"), default="auto")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--system_prompt", default=None)

    parser.add_argument(
        "--question_fields", default="question,problem,prompt,input,extra_info.problem"
    )
    parser.add_argument(
        "--answer_fields", default="answer,final_answer,reward_model.ground_truth"
    )
    parser.add_argument(
        "--response_fields",
        default="model_output,output,response,llm_response,prediction",
    )
    parser.add_argument(
        "--label_fields",
        default="final_correct,math_verify_ok,correct,is_correct,label,score,math_verify_score",
    )

    parser.add_argument(
        "--reference_file",
        "--solution_file",
        dest="reference_files",
        action="append",
        help="Original dataset for solution lookup; may be repeated.",
    )
    parser.add_argument(
        "--reference_input_format",
        "--solution_input_format",
        dest="reference_input_format",
        choices=("auto", "json", "jsonl"),
        default="auto",
    )
    parser.add_argument(
        "--reference_question_fields",
        default="question,problem,prompt,input,extra_info.problem",
    )
    parser.add_argument(
        "--reference_solution_fields",
        "--solution_fields",
        "--solution_field",
        dest="reference_solution_fields",
        default="solution,extra_info.solution,response_content",
    )
    parser.add_argument(
        "--missing_solution",
        choices=("error", "skip"),
        default="error",
    )
    parser.add_argument(
        "--multi_solution_mode",
        choices=MULTI_SOLUTION_MODES,
        default="first",
        help=(
            "How to handle multiple matched solutions: 'first' uses only the first; "
            "'all' puts all solutions into one prompt."
        ),
    )

    parser.add_argument("--balance", choices=("upsample", "none"), default="upsample")
    parser.add_argument(
        "--all_correct_downsample_factor",
        type=int,
        default=4,
        help="Keep approximately 1/factor responses for all-correct questions; 1 disables.",
    )
    parser.add_argument(
        "--balance_seed", "--filter_seed", dest="balance_seed", type=int, default=42
    )
    parser.add_argument("--drop_empty_response", action="store_true", default=True)
    parser.add_argument(
        "--keep_empty_response", action="store_false", dest="drop_empty_response"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None, default_mode: str = "raw") -> int:
    args = build_parser(default_mode).parse_args(argv)
    input_path = Path(args.input_file).resolve()
    output_path = Path(args.output_file).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output file already exists: {output_path}; use --overwrite")

    question_fields = parse_csv_fields(args.question_fields)
    answer_fields = parse_csv_fields(args.answer_fields)
    response_fields = parse_csv_fields(args.response_fields)
    label_fields = parse_csv_fields(args.label_fields)
    reference_question_fields = parse_csv_fields(args.reference_question_fields)
    reference_solution_fields = parse_csv_fields(args.reference_solution_fields)
    for name, fields in (
        ("question_fields", question_fields),
        ("answer_fields", answer_fields),
        ("response_fields", response_fields),
        ("label_fields", label_fields),
    ):
        if not fields:
            raise ValueError(f"--{name} is empty")

    candidates, input_stats = collect_candidates(
        path=input_path,
        input_format=args.input_format,
        mode=args.mode,
        question_fields=question_fields,
        answer_fields=answer_fields,
        response_fields=response_fields,
        label_fields=label_fields,
        drop_empty_response=args.drop_empty_response,
    )
    if not candidates:
        raise RuntimeError("no usable input rows")

    candidates, downsample_stats = downsample_all_correct_questions(
        candidates,
        factor=args.all_correct_downsample_factor,
        seed=args.balance_seed,
    )
    input_stats.update(downsample_stats)

    solution_map: Dict[str, List[str]] = {}
    if args.mode == "solution":
        if not args.reference_files:
            raise ValueError("--reference_file is required for --mode=solution")
        reference_paths = [Path(path).resolve() for path in args.reference_files]
        missing_files = [str(path) for path in reference_paths if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(f"reference files not found: {missing_files}")
        solution_map = load_solution_index(
            paths=reference_paths,
            input_format=args.reference_input_format,
            question_fields=reference_question_fields,
            solution_fields=reference_solution_fields,
            needed_questions={candidate.question for candidate in candidates},
        )
        missing_questions = sorted(
            {candidate.question for candidate in candidates if candidate.question not in solution_map}
        )
        if missing_questions and args.missing_solution == "error":
            examples = "\n".join(f"  - {question[:200]!r}" for question in missing_questions[:5])
            raise ValueError(
                f"{len(missing_questions)} questions have no exact solution match. "
                "Use --missing_solution=skip to skip them. Examples:\n"
                f"{examples}"
            )
        if missing_questions:
            missing_set = set(missing_questions)
            before = len(candidates)
            candidates = [item for item in candidates if item.question not in missing_set]
            input_stats["skipped_missing_solution"] = before - len(candidates)
            if not candidates:
                raise RuntimeError("no candidates remain after missing-solution filtering")

    if args.balance == "upsample":
        extra_repetitions, original_counts = plan_upsampling(
            candidates, args.balance_seed
        )
    else:
        extra_repetitions = Counter()
        original_counts = output_class_counts(candidates)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("tokenizer does not support apply_chat_template")
    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError("tokenizer.chat_template is empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_counts = write_output(
        output_path=output_path,
        candidates=candidates,
        mode=args.mode,
        solution_map=solution_map,
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        extra_repetitions=extra_repetitions,
        multi_solution_mode=args.multi_solution_mode,
    )

    print(f"mode={args.mode}")
    print(f"input_file={input_path}")
    if args.mode == "solution":
        print(f"reference_files={args.reference_files}")
        print(f"multi_solution_mode={args.multi_solution_mode}")
    print(f"output_file={output_path}")
    for key in sorted(input_stats):
        print(f"{key}={input_stats[key]}")
    print(f"original_positive={original_counts[1.0]}")
    print(f"original_negative={original_counts[0.0]}")
    print(f"upsampled_copies={sum(extra_repetitions.values())}")
    print(f"written_positive={written_counts[1.0]}")
    print(f"written_negative={written_counts[0.0]}")
    print(f"written_rows={sum(written_counts.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
