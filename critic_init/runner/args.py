from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to the pretrained model or model identifier."}
    )

    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Allow custom model/tokenizer code from the model repo."},
    )

    train_value_head_only: bool = field(
        default=False,
        metadata={
            "help": "Freeze the language-model backbone and train only the value head."
        },
    )


@dataclass
class DataTrainingArguments:
    train_file: str = field(
        metadata={"help": "Path to the training jsonl file."}
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap on the number of training samples."},
    )

    max_length: int = field(
        default=16000,
        metadata={"help": "Maximum tokenized sequence length used for training."},
    )

    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of processes used to tokenize and build response masks once before training."
        },
    )

    preprocessing_batch_size: int = field(
        default=256,
        metadata={"help": "Batch size used by the one-time tokenization pass."},
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Rebuild the cached tokenized training dataset."},
    )

    use_length_grouping: bool = field(
        default=True,
        metadata={
            "help": "Group examples of similar token length to reduce padding in each batch."
        },
    )
