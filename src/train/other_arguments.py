from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments, Seq2SeqTrainingArguments

@dataclass
class otherArguments:
    sl_max_len: int = field(
        default=20,
        metadata={
            "help": "The max length of label"}
    )
    teacher_model_name_or_path: str = field(
        default=None,
        metadata={"help": "teacher_model_path"}
    )

    num_prompt: int = field(
        default=10,
        metadata={"help": "number of the prompt"}
    )

    fixed_decoder: bool = field(
        default=True,
        metadata={"help": "if fix decoder"}
    )

    train_LM_out_label:bool = field(
        default=True,
        metadata={"help": "use KL"}
    )

    # task_prompt:bool = field(
    #     default=False,
    #     metadata={"help": "train task prompt"}
    # )


