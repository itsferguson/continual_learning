from dataclasses import dataclass, field
import dataclasses
from typing import Optional, List, Tuple
from transformers import AdapterArguments, Seq2SeqTrainingArguments

ANS_TOKEN = "ANS"

INPUT_TOKENS = ["input1", "input2"]


def gen_token(dataset_name):
    return f"{dataset_name}_GEN"


@dataclass
class DataArguments:
    datasets: List[str] = field(
        default_factory=list,
        metadata={"help": "a list of strings that correspond with some datasets"},
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "limits the number of training steps"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "limits the number of training steps"},
    )
    replay_rate: float = field(
        default=0.05,
        metadata={
            "help": "Rate of samples from previous datasets to augment the current dataset with.\
                Integer values correspond with an exact number of sample.\
                Float values are a percentage compared to the number of samples in the current datset"
        },
    )
    sampling: str = field(
        default="none",
        metadata={"help": "Whether of not to use pseudo samples for regularization"},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    prefix: str = field(
        default="",
        metadata={
            "help": 'Prefix like "summarize:". Look at T5 Paper for more information.'
        },
    )
    data_folder: str = field(
        default="./data", metadata={"help": "folder containing the datasets"}
    )

    def __post_init__(self):
        if self.replay_rate == 0:
            self.sampling = "none"

        if self.sampling == "none":
            self.replay_rate = 0
            self.real_sampling = False
            self.pseudo_sampling = False
        elif self.sampling == "real":
            self.real_sampling = True
            self.pseudo_sampling = False
        elif self.sampling == "pseudo":
            self.real_sampling = False
            self.pseudo_sampling = True


@dataclass
class CustomAdapterArguments(AdapterArguments):
    adapter_config: str = field(default="none")

    def __post_init__(self):
        if self.adapter_config == "none":
            self.train_adapter = False
        else:
            self.train_adapter = True


@dataclass
class Config:
    task_name: str = field(
        metadata={
            "help": "will corrispond to the name of the adapter if adapter training is activated"
        },
    )
    name: Optional[str] = field(
        default=None, metadata={"help": "set the name of the run"}
    )
    model_type: str = field(default="t5-small", metadata={"help": ""})


@dataclass
class CustomTrainingArguments(Seq2SeqTrainingArguments):
    lm_lambda: float = field(
        default=1.0,
        metadata={
            "help": "Scales how much the influence the Language Model Loss has over the total loss"
        },
    )
    metric: str = field(
        default="rouge",
        metadata={"help": "corresponds to a huggingface evaluate metric"},
    )
    output_dir: str = field(
        default="./trainer",
        metadata={
            "help": "this where data like generated samples and trained models will be stored"
        },
    )
    predict_with_generate: bool = True
    evaluation_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch")
    optim: str = field(default="adafactor")
