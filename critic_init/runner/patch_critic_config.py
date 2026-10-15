#!/usr/bin/env python
# coding=utf-8
"""Backfill the `architectures` field into a saved critic's config.json.

trl's AutoModelForCausalLMWithValueHead wraps the base model at runtime and does
not write `architectures` back to config.json on save. verl's HFModelConfig
asserts on exactly one architecture, so loading such a critic fails with
"Expect only one architecture, got None". Run this once against an
already-trained critic directory to fix it without retraining.

Usage:
    python patch_critic_config.py <critic_ckpt_dir> [--base <base_model_path_or_name>]

If --base is omitted, you must pass --arch to specify the architecture explicitly,
e.g. --arch Qwen2ForCausalLM
"""

import argparse
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser(description="Patch critic config.json architectures field.")
    parser.add_argument("critic_dir", help="Directory containing the saved critic config.json.")
    parser.add_argument(
        "--base",
        default=None,
        help="Base model path or HF name to read architectures from.",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="Explicit architecture name to write, e.g. Qwen2ForCausalLM. "
        "Takes precedence over --base.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite architectures even if it already exists.",
    )
    args = parser.parse_args()

    config_path = os.path.join(args.critic_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"No config.json found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    existing = config_dict.get("architectures", None)
    if existing and not args.force:
        print(f"config.json already has architectures: {existing}. Use --force to overwrite.")
        return

    # Resolve target architectures.
    if args.arch:
        architectures = [args.arch]
    elif args.base:
        from transformers import AutoConfig

        base_config = AutoConfig.from_pretrained(args.base)
        architectures = getattr(base_config, "architectures", None)
        if not architectures:
            raise ValueError(
                f"Base model config ({args.base}) has no architectures field. "
                "Pass --arch explicitly instead."
            )
    else:
        raise ValueError("You must provide either --base or --arch to determine architectures.")

    # Backup then write.
    backup_path = config_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(config_path, backup_path)
        print(f"Backed up original config to {backup_path}")

    config_dict["architectures"] = architectures
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"Patched {config_path} with architectures: {architectures}")


if __name__ == "__main__":
    main()
