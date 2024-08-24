import torch
import os
import numpy as np
import dataset
from dataset_properties import adjust_k_per_class
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer


class Trainer:
    def __init__(
        self,
        model_name,
        task_list,
        language_modeling=False,
        memory_perc=0,
        batch_size=8,
        select_k_per_class=-1,
        lr=0.3,
        weight_decay=1e-5,
        seq_len=512,
        early_stopping=True,
        get_test_subset=True,
    ):

        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        self.task_list = task_list
        self.language_modeling = language_modeling
        self.memory_perc = memory_perc
        self.tasks_data_dict = self.get_tasks_data_dict()

        self.batch_size = batch_size
        self.select_k_per_class = select_k_per_class
        self.lr = lr
        self.weight_decay = weight_decay
        self.seq_len = seq_len
        self.early_stopping = early_stopping
        self.get_test_subset = get_test_subset

    # Create optimizer
    def get_optimizer(self):
        lr = self.lr
        weight_decay = self.weight_decay

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": lr,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
        return optimizer

        # Create Dictionary of task_name -> dataloader (for CL experiments)

    def get_tasks_data_dict(self):
        tasks_data_dict = {}
        ds = dataset.T5Dataset(self.tokenizer)

        for task in self.task_list:
            tasks_data_dict[task] = {}
            print(task)
            data_params = {
                "task": task,
                "batch_size": self.batch_size,
                "max_length": self.seq_len,
                "prefix_list": [],  # we are using vector prefix (instead of tokenization)
            }
            k, k_val = adjust_k_per_class(
                task, self.select_k_per_class, self.get_test_subset
            )

            dataloader_train = ds.get_dataset(**data_params, k=k, split="train")
            tasks_data_dict[task]["train"] = dataloader_train
            if self.language_modeling is True:
                tasks_data_dict[task]["train_lm"] = ds.get_dataset(
                    **data_params, k=k, split="train", lm_task=True
                )

            print("k = ", k, "  k-val = ", k_val)

            # TODO do this after training the task
            # because we first need to train the language_model
            # if memory_perc > 0:
            #    k_mem = max(
            #        1, int(len(dataloader_train) * self.batch_size * memory_perc)
            #    )
            #    if self.lm_
            #    dataloader_mem = ds2.get_final_ds(**data_params, k=k_mem, split="train")
            #    tasks_data_dict[task]["train_mem"] = dataloader_mem

            dataloader_val = ds.get_dataset(**data_params, k=k_val, split="validation")
            tasks_data_dict[task]["val"] = dataloader_val

            if self.get_test_subset:
                dataloader_test = ds.get_dataset(**data_params, k=k_val, split="test")
                tasks_data_dict[task]["test"] = dataloader_test

            if task == "multirc" and k_val == -1:
                self.multirc_idx = (
                    ds2.multirc_idx
                )  # saving multirc idx for later computation
            else:
                self.multirc_idx = None

        return tasks_data_dict

    # TODO
    def train_one_task(
        self,
        task,
        epochs=40,
        progressive=True,
        eval_every_N=1,
        eval_on_all_tasks=False,
        data_replay_freq=-1,
    ):

        print("task = ", task)
        if self.early_stopping:
            self.best_acc = 0.0  # re-setting best acc

        model = self.model
        model.to(self.device)

        with torch.no_grad():
            self.optimizer = self.get_optimizer()

        dataloader_train = self.tasks_data_dict[task]["train"]
        if self.language_modeling:
            dataloader_train_lm = self.tasks_data_dict[task]["train_lm"]
            dataloader_train = zip(dataloader_train, dataloader_train_lm)
        dataloader_val = self.tasks_data_dict[task]["val"]

        val_acc = []

        for epoch in range(epochs):
            print(f"Epoch ({epoch}/{epochs})")

            model.train()

            for i, batch in enumerate(tqdm(dataloader_train)):
                if self.language_modeling:
                    batch, lm_batch = batch
                    batch_lm = {k: lm_batch[k].to("cuda") for k in lm_batch}
                    lm_loss = self.train_step(batch_lm)
                else:
                    lm_loss = 0

                batch = {k: batch[k].to("cuda") for k in batch}
                task_loss = self.train_step(batch)

                loss = task_loss + self.lm_lamba * lm_loss

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # performing data replay on all previous tasks
                if data_replay_freq != -1 and i % data_replay_freq == 0:
                    self.memory_replay(tasks_to_generators, progressive)

            if epoch % eval_every_N == 0:
                overall_acc = []
                if eval_on_all_tasks:
                    # eval current model/prompt on all tasks (for approaches that suffer from catastrophic forgetting)
                    for eval_task in self.task_list:
                        acc = self.validate(
                            self.tasks_data_dict[eval_task]["val"],
                            eval_task,
                            prompt=prompt,
                            target_len=self.task_to_target_len[eval_task],
                            print_outputs=False,
                        )
                        overall_acc.append(np.mean(acc))
                        if (
                            eval_task == task
                        ):  # record val accuracy for the current task
                            val_acc.append(np.mean(acc))
                    acc = np.mean(overall_acc)
                else:
                    acc = self.validate(
                        dataloader_val,
                        task,
                        prompt=prompt,
                        target_len=target_len,
                        print_outputs=True,
                    )
                    if task in ["record", "cb"] or (
                        task == "multirc" and self.multirc_idx != None
                    ):
                        acc = np.mean(acc)  # averaging 2 scores
                    val_acc.append(acc)

                if self.early_stopping:
                    self.update_best_model(acc, task=task)
                print(epoch, task, "->", val_acc[-1])

        if progressive:
            self.progress_previous_prompts(task=task)

        else:
            if self.early_stopping:
                self.restore_best_model()
        return val_acc

    # TODO
    def validate():
        pass

    # Train model continually
    def train_continual(
        self,
        task_list,
        epochs=40,
        save_path=None,
        eval_every_N=1,
        test_eval_after_every_task=False,  # only needed for methods with catastrophic forgetting
        data_replay_freq=-1,
    ):
        results_dict = {}
        if self.get_test_subset:
            results_dict["test"] = {}

        for num, task in enumerate(task_list):
            eval_on_all_tasks = False if len(task_list) == 1 else True
            eval_frq = eval_every_N if not eval_on_all_tasks else int(epochs // 3)
            val_acc = self.train_one_task(
                task,
                epochs,
                eval_every_N=eval_frq,
                data_replay_freq=data_replay_freq,
                eval_on_all_tasks=eval_on_all_tasks,
            )

            print(task, val_acc)
            results_dict[task] = val_acc

            print("Calculating test acc ...")
            if self.get_test_subset:
                if test_eval_after_every_task:
                    # eval test accuracy for all tasks
                    results_dict["test"][num] = {}
                    for test_task in task_list:
                        acc = self.validate(
                            self.tasks_data_dict[test_task]["test"],
                            test_task,
                            self.task_to_target_len[test_task],
                            print_outputs=True,
                        )
                        results_dict["test"][num][test_task] = acc

                else:
                    acc = self.validate(
                        self.tasks_data_dict[task]["test"],
                        task,
                        self.task_to_target_len[task],
                        print_outputs=True,
                    )
                    results_dict["test"][task] = acc
            # saving results dict after each task
            np.save(os.path.join(save_path, "results_dict.npy"), results_dict)

        return results_dict
