import datasets
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, T5Tokenizer

from dataset_properties import (
    get_input_keys,
    get_label_key,
    get_labels,
    has_non_discrete_labels,
    is_glue_dataset,
    is_local_dataset,
    is_superglue_dataset,
    has_dedicated_val_split,
)


class T5Dataset:
    def __init__(self, tokenizer: T5Tokenizer, seed=42):
        """Dataset class for T5 model experiments.
        Args:
            task (str): Name of the downstream task.
            tokenizer (HuggingFace Tokenizer): T5 model tokenizer to use.
        """
        self.tokenizer = tokenizer
        self.collate_fn = DataCollatorForSeq2Seq(tokenizer, padding=True)
        self.seed = seed
        np.random.seed(seed)

    # Helper function to select a subset of k samples per class in a dataset
    def select_subset(self, task, dataset: Dataset, k) -> Dataset:
        if has_non_discrete_labels(task):  # non-discrete labels
            selected_indices = np.random.choice(
                np.arange(dataset.shape[0]), min(k, dataset.shape[0]), replace=False
            )

        else:
            label_key = get_label_key(task)
            labels = set(dataset[label_key])
            selected_indices = np.array([], dtype="int64")

            for label in labels:
                current_indices = np.where(np.array(dataset[label_key]) == label)[0]
                selected_indices = np.concatenate(
                    [
                        selected_indices,
                        np.random.choice(
                            current_indices,
                            min(k, current_indices.shape[0]),
                            replace=False,
                        ),
                    ]
                )

        np.random.seed(self.seed)
        np.random.shuffle(selected_indices)
        return dataset.select(selected_indices)

    # Function to preprocess raw input & label text into tokenized dictionary
    def encoding_function(self, examples, lm_task, max_length):
        tokenizer = self.tokenizer
        if lm_task:
            labels = [f"ANS {label}" for label in examples["text_label"]]
            source = tokenizer(examples["gen_token"])
            target = tokenizer(
                examples["text_input"],
                labels,
                truncation="only_first",
                max_length=max_length,
            )
        else:
            source = tokenizer(
                examples["text_input"],
                truncation=True,
                max_length=max_length,
            )
            target = tokenizer(examples["text_label"])

        dict_final = {
            "input_ids": source["input_ids"],
            "attention_mask": source["attention_mask"],
            "labels": target["input_ids"],
        }
        return dict_final

    # A wrapper around datasets.load_dataset to handle all special cases
    def load_dataset(self, task, split) -> Dataset:
        if is_local_dataset(task):
            df = pd.read_csv(f"../datasets/src/data/{task}/{split}.csv", header=None)
            df = df.rename(columns={0: "label", 1: "title", 2: "content"})
            df["label"] = df["label"] - 1
            dataset = datasets.Dataset.from_pandas(df)
        elif task == "mnli":
            if split == "train":
                dataset = load_dataset("glue", "mnli", split="train")
            else:
                # load combined matched and mismatched validation sets
                dataset = load_dataset(
                    "LysandreJik/glue-mnli-train", split="validation"
                )
        elif task == "stsb":
            dataset = load_dataset(
                "stsb_multi_mt",
                name="en",
                split=split if split != "validation" else "dev",
            )
        elif is_glue_dataset(task):
            dataset = load_dataset(
                "glue",
                task,
                split="validation" if split == "test" else split,
            )
        elif is_superglue_dataset(task):
            dataset = load_dataset(
                "super_glue",
                task.replace("_superglue", "").replace("_bool", ""),
                split="validation" if split == "test" else split,
            )
        else:
            dataset = load_dataset(task, split="train" if split == "train" else "test")

        # removes lsp warnings
        assert isinstance(dataset, Dataset)

        # if split is not train and
        # validation split is missing
        # split test set in half
        if split != "train" and not has_dedicated_val_split(task):
            datadict = dataset.train_test_split(test_size=0.5, seed=self.seed)
            if split == "test":
                dataset = datadict["test"]  # test split
            else:
                dataset = datadict["train"]  # validation split

        return dataset

    def preprocess_dataset(self, task, dataset: Dataset) -> Dataset:
        """Function that does some further processing on the dataset depending on the current downstream
        task.
        args:
            task(str): Name of the downstream task.
            dataset(Dataset): Dataset object containing the raw data.
        """
        # For yahoo dataset we need to filter out empty rows
        # (i.e. where "question" field is empty)
        if task == "yahoo_answers_topics":
            # TODO filter out empty rows
            pass

        # Using Lester et al. setting for WSC task, e.g.
        # using only positive samples (for output generation)
        if task == "wsc":
            idx = np.where(np.array(dataset["label"]) == 1)[0]
            dataset = dataset.select(idx)

        return dataset

    def add_missing_columns(self, example, task):
        label_key = get_label_key(task)
        input_keys = get_input_keys(task)
        if len(input_keys) > 1:
            text = ""
            for key in input_keys:
                text += key + ": " + str(example[key]) + " "
        else:
            text = example[input_keys[0]]

        text = text.strip()

        example["gen_token"] = f"GEN_{task}"
        example["text_input"] = text
        example["text_label"] = get_labels(task)[example[label_key]]

        return example

    def get_dataset(
        self,
        task,
        split,
        batch_size,
        k=-1,
        max_length=512,
        lm_task=False,
    ):
        """Function that returns final T5 dataloader.
        args:
            task(str): Name of the downstream task.
            split(str): Which data split to use(train / validation / test).
            batch_size(int): Batch size to use in the dataloader.
            k(int, optional): Number of samples to use for each class . Defaults to - 1, not sub - sample the data.
            return_test(bool, optional): Whether to create a test split.
                When True, two Dataloaders are returned. Defaults to False.
            target_len (int, optional): Length of the model output (in tokens). Defaults to 2.
            max_length (int, optional): Length of the model input (in tokens). Defaults to 512.
            lm_task (bool, optional): Whether or not to format the dataset for the language modeling
                task.

        returns:
            Dataloader: Torch Dataloader with preprocessed input text & label.
        """

        # Load the dataset from HuggingFace or csv file
        dataset = self.load_dataset(task, split)

        # Do some task specific preprocessing
        dataset = self.preprocess_dataset(task, dataset)

        # Selecting k subset of the samples (if requested)
        if k != -1:
            dataset = self.select_subset(task, dataset, k=k)

        # Shuffle the dataset
        dataset = dataset.shuffle(seed=self.seed)

        # Generate Misisng Columns
        dataset = dataset.map(
            lambda x: self.add_missing_columns(x, task=task), batched=False
        )

        # Tokenize the dataset
        encoded_dataset = dataset.map(
            lambda x: self.encoding_function(x, max_length=max_length, lm_task=lm_task),
            batched=True,
        )

        encoded_dataset = encoded_dataset.select_columns(
            ["input_ids", "attention_mask", "labels"]
        )

        return DataLoader(
            encoded_dataset, collate_fn=self.collate_fn, batch_size=batch_size
        )
