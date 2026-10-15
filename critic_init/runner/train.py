#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
import numpy as np
import json
import inspect
import re
import copy
import math
import random
from pathlib import Path

random.seed(42)

import torch
import transformers
from args import DataTrainingArguments, ModelArguments
from pretrainer import PreTrainer
from response_mask import tokenize_with_response_mask

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding,
)

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None

from transformers.trainer_utils import is_main_process
from transformers.utils import check_min_version
from datasets import load_dataset
from typing import Any, Dict, List


def is_trl_available():
    try:
        import trl  # noqa: F401
        return True
    except ImportError:
        return False


def patch_valuehead_model(model):
    """Patch the value head model so its forward returns logits compatible with TokenClassification."""
    original_forward = model.forward

    def patched_forward(self, *args, **kwargs):
        output = original_forward(*args, **kwargs)
        # AutoModelForCausalLMWithValueHead returns (lm_logits, loss, value)
        # We need to return an object with .logits = value for our trainer
        from transformers.modeling_outputs import TokenClassifierOutput
        value = output[2].unsqueeze(-1)
        return TokenClassifierOutput(logits=value)

    import types
    model.forward = types.MethodType(patched_forward, model)


def load_valuehead_model(local_path, torch_dtype, model_config):
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=local_path,
            torch_dtype=torch_dtype,
            config=model_config,
            attn_implementation="flash_attention_2",
        )
        return model
    except BaseException as e:
        if not is_trl_available():
            raise RuntimeError(
                f"model({local_path}) is not a value head model, please install trl to make it valid"
            ) from e

    assert is_trl_available()

    from trl import AutoModelForCausalLMWithValueHead

    if (
        AutoModelForVision2Seq is not None
        and type(model_config) in AutoModelForVision2Seq._model_mapping.keys()
    ):
        module_class = AutoModelForVision2Seq
    else:
        module_class = AutoModelForCausalLM

    ori_model = module_class.from_pretrained(
        pretrained_model_name_or_path=local_path,
        torch_dtype=torch_dtype,
        config=model_config,
        attn_implementation="flash_attention_2",
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(ori_model)
    patch_valuehead_model(model)
    return model


VALUE_HEAD_ATTRIBUTE_CANDIDATES = (
    "v_head",  # TRL AutoModelForCausalLMWithValueHead
    "value_head",
    "score",
    "classifier",
    "classification_head",
)


def configure_trainable_parameters(model, train_value_head_only):
    """Optionally freeze everything except the scalar value-head module."""
    if not train_value_head_only:
        logger.info("Training the full critic model.")
        return

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trainable_parameter_ids = set()
    matched_head_attributes = []
    for attribute_name in VALUE_HEAD_ATTRIBUTE_CANDIDATES:
        head_module = getattr(model, attribute_name, None)
        if not isinstance(head_module, torch.nn.Module):
            continue

        head_parameters = list(head_module.parameters())
        if not head_parameters:
            continue
        matched_head_attributes.append(attribute_name)
        for parameter in head_parameters:
            parameter.requires_grad_(True)
            trainable_parameter_ids.add(id(parameter))

    if not trainable_parameter_ids:
        available_children = [name for name, _ in model.named_children()]
        raise ValueError(
            "--train_value_head_only was enabled, but no supported value-head module "
            f"was found. Expected one of {VALUE_HEAD_ATTRIBUTE_CANDIDATES}; "
            f"top-level modules are {available_children}."
        )

    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in trainable_parameter_ids
    ]
    logger.info(
        "Value-head-only training enabled; trainable head(s): %s; parameters: %s",
        matched_head_attributes,
        trainable_names,
    )



# Will error if the minimal version of Transformers is not installed.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)

CHECKPOINT_DIR_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def get_num_params(model):
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()

        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params

        if param.requires_grad:
            trainable_params += num_params

    logger.info(
        f"trainable params: {int(trainable_params / 1e6)}M || "
        f"all params: {int(all_param / 1e6)}M || "
        f"trainable: {100 * trainable_params / all_param}%"
    )


class RewardDataCollator:

    def __init__(self, tokenizer):
        # 使用标准的 DataCollatorWithPadding 处理 padding
        self.tokenizer = tokenizer
        self.base_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest"
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # Custom fields must be padded separately. response_mask uses zero for
        # both prompt tokens and padding tokens.
        model_features = []
        response_masks = []
        scores_list = []
        for feature in features:
            model_feature = dict(feature)
            response_masks.append(list(model_feature.pop("response_mask")))
            scores_list.append(model_feature.pop("scores"))
            # Used by Trainer's LengthGroupedSampler, not by the model/collator.
            model_feature.pop("length", None)
            model_features.append(model_feature)

        batch = self.base_collator(model_features)

        padded_length = batch["input_ids"].shape[1]

        # 取出 scores
        scores = torch.tensor(
            scores_list,
            dtype=torch.float32
        )

        # (batch) -> (batch,1) -> (batch,padded_length)
        labels = scores.unsqueeze(1).repeat(1, padded_length)

        batch["labels"] = labels

        padded_response_masks = []
        for response_mask in response_masks:
            padding_length = padded_length - len(response_mask)
            if padding_length < 0:
                raise ValueError("response_mask is longer than the padded input")
            padding = [0] * padding_length
            if self.tokenizer.padding_side == "left":
                response_mask = padding + response_mask
            else:
                response_mask = response_mask + padding
            padded_response_masks.append(response_mask)
        batch["response_mask"] = torch.tensor(padded_response_masks, dtype=torch.long)

        return batch


def patch_saved_config_architectures(output_dir, base_model_name_or_path):
    """Ensure the saved config.json carries an `architectures` field.

    trl's AutoModelForCausalLMWithValueHead wraps the base model at runtime and
    does not write `architectures` back to config.json on save. verl's
    HFModelConfig.__post_init__ asserts on exactly one architecture, so a missing
    field raises "Expect only one architecture, got None". We backfill it from the
    base model's config.
    """
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.isfile(config_path):
        logger.warning("No config.json found at %s, skip architectures patch.", config_path)
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    architectures = config_dict.get("architectures", None)
    if architectures:
        logger.info("config.json already has architectures: %s", architectures)
        return

    base_config = AutoConfig.from_pretrained(base_model_name_or_path)
    base_architectures = getattr(base_config, "architectures", None)
    if not base_architectures:
        raise ValueError(
            f"Base model config ({base_model_name_or_path}) has no architectures field; "
            "cannot backfill the saved critic config.json automatically."
        )

    config_dict["architectures"] = base_architectures
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    logger.info("Patched %s with architectures: %s", config_path, base_architectures)


def get_latest_resumable_checkpoint(output_dir):
    checkpoint_dirs = []
    invalid_checkpoints = []

    for entry in Path(output_dir).iterdir():
        if not entry.is_dir():
            continue

        match = CHECKPOINT_DIR_PATTERN.match(entry.name)
        if match is None:
            continue

        checkpoint_dirs.append((int(match.group(1)), entry))

    checkpoint_dirs.sort(key=lambda item: item[0], reverse=True)

    for _, checkpoint_dir in checkpoint_dirs:
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        if trainer_state_path.is_file():
            return str(checkpoint_dir), invalid_checkpoints

        invalid_checkpoints.append(str(checkpoint_dir))

    return None, invalid_checkpoints


def main():

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):

        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )

    else:

        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detect the latest resumable checkpoint and skip incomplete ones left
    # behind by interrupted saves.
    last_checkpoint = None
    invalid_checkpoints = []

    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):

        last_checkpoint, invalid_checkpoints = get_latest_resumable_checkpoint(
            training_args.output_dir
        )

        for checkpoint_dir in invalid_checkpoints:
            logger.warning(
                "Ignoring incomplete checkpoint without trainer_state.json: %s",
                checkpoint_dir,
            )

        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            if invalid_checkpoints:
                logger.warning(
                    "No valid checkpoint found under %s. Training will restart from scratch.",
                    training_args.output_dir,
                )
            else:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )

        elif last_checkpoint is not None:

            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. "
                "To avoid this behavior, change the --output_dir or add --overwrite_output_dir."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank)
        else logging.WARN
    )

    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )

    if is_main_process(training_args.local_rank):

        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed
    set_seed(training_args.seed)

    def tokenize_batch(batch):

        tokenized_output = tokenize_with_response_mask(
            tokenizer=tokenizer,
            prompts=batch["prompt"],
            responses=batch.get("response"),
            max_length=data_args.max_length,
        )

        tokenized_output["scores"] = batch["label"]
        tokenized_output["length"] = [
            len(input_ids) for input_ids in tokenized_output["input_ids"]
        ]

        return tokenized_output

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("critic response-mask training requires a fast tokenizer with offset mappings")

    training_args.remove_unused_columns = False

    # Tokenize once instead of repeating tokenizer/offset work in every epoch.
    # main_process_first lets one process per node populate the shared Arrow
    # cache while the other local ranks wait and then reuse it.
    with training_args.main_process_first(desc="tokenize critic training data"):
        train_dataset = load_dataset(
            "json",
            data_files={"train": data_args.train_file},
            split="train",
        )
        raw_columns = train_dataset.column_names
        train_dataset = train_dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=data_args.preprocessing_batch_size,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_columns,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing and building response masks",
        )

    print("数据集大小:", len(train_dataset))

    if data_args.use_length_grouping:
        training_args.length_column_name = "length"
        if hasattr(training_args, "train_sampling_strategy"):
            # Transformers 5.x
            training_args.train_sampling_strategy = "group_by_length"
        elif hasattr(training_args, "group_by_length"):
            # Transformers 4.x
            training_args.group_by_length = True
        else:
            raise RuntimeError(
                "this Transformers version does not expose a length-grouped training sampler"
            )
        logger.info("Length-grouped batching enabled using the cached 'length' column.")

    data_collator = RewardDataCollator(tokenizer=tokenizer)

    # ================= 加载模型 =================

    critic_model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
    )

    critic_model_config.num_labels = 1
    critic_model_config.use_cache = False

    torch_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    model = load_valuehead_model(
        local_path=model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        model_config=critic_model_config,
    )
    configure_trainable_parameters(model, model_args.train_value_head_only)
    model.config.use_cache = False
    if hasattr(model, "pretrained_model"):
        model.pretrained_model.config.use_cache = False
    model = model.to("cuda")

    get_num_params(model)

    # Initialize Trainer

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    if "processing_class" in inspect.signature(transformers.Trainer.__init__).parameters:
        # Transformers 5.x renamed the Trainer tokenizer argument.
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = PreTrainer(**trainer_kwargs)

    if training_args.do_train:

        checkpoint = last_checkpoint if last_checkpoint else None

        train_result = trainer.train(
            resume_from_checkpoint=checkpoint
        )

        metrics = train_result.metrics

        trainer.save_model()

        # trl's value-head wrapper does not persist `architectures` to config.json.
        # Backfill it so verl can load this critic without hitting
        # "Expect only one architecture, got None".
        if is_main_process(training_args.local_rank):
            patch_saved_config_architectures(
                training_args.output_dir,
                model_args.model_name_or_path,
            )

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )

        metrics["train_samples"] = min(
            max_train_samples,
            len(train_dataset)
        )

        trainer.log_metrics("train", metrics)

        trainer.save_metrics("train", metrics)

        trainer.save_state()


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
