import copy
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import wandb
from loguru import logger
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torchvision import transforms
from tqdm import tqdm

from src.datasets.helper import DatasetName, get_dataset
from src.models.embedder import Embedder
from src.models.helper import embed_dataset
from src.trainers.eval_types.base import BaseEvalType
from src.trainers.eval_types.bias_evaluation import EvalBias
from src.trainers.eval_types.dummy_classifier import (
    EvalDummyConstant,
    EvalDummyMostFrequent,
    EvalDummyUniform,
)
from src.trainers.eval_types.fine_tuning import EvalFineTuning
from src.trainers.eval_types.knn import EvalKNN
from src.trainers.eval_types.lin import EvalLin
from src.utils.utils import fix_random_seeds

eval_type_dict = {
    # Baselines
    # "dummy_most_frequent": EvalDummyMostFrequent,
    # "dummy_uniform": EvalDummyUniform,
    # "dummy_constant": EvalDummyConstant,
    # Models
    # "fine_tuning": EvalFineTuning,
    "bias_evaluation": EvalBias,
}


class BiasEvaluationTrainer(ABC, object):
    def __init__(
        self,
        dataset_name: DatasetName,
        config: dict,
        SSL_model: str = "imagenet",
        output_path: Union[Path, str] = "assets/evaluation",
        cache_path: Union[Path, str] = "assets/evaluation/cache",
        n_layers: int = 1,
        append_to_df: bool = False,
        wandb_project_name: str = "PASSION-Evaluation",
        log_wandb: bool = False,
    ):
        self.dataset_name = dataset_name
        self.config = config
        self.output_path = Path(output_path)
        self.cache_path = Path(cache_path)
        self.append_to_df = append_to_df
        self.wandb_project_name = wandb_project_name
        self.log_wandb = log_wandb
        self.seed = config["seed"]
        fix_random_seeds(self.seed)

        self.df_name = f"{self.experiment_name}_{self.dataset_name.value}.csv"
        self.df_path = self.output_path / self.df_name
        self.model_path = self.output_path / self.experiment_name

        # make sure the output and cache path exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)

        # parse the config to get the eval types
        self.eval_types = []
        for k, v in self.config.items():
            if k in eval_type_dict.keys():
                self.eval_types.append((eval_type_dict.get(k), v))

        # save the results to the dataframe
        self.df = pd.DataFrame(
            [],
            columns=[
                "Score",
                "EvalTargets",
                "EvalPredictions",
                "EvalType",
                "AdditionalRunInfo",
                "SplitName",
            ],
        )
        if append_to_df:
            if not self.df_path.exists():
                print(f"Results for dataset: {self.dataset_name.value} not available.")
            else:
                print(f"Appending results to: {self.df_path}")
                self.df = pd.read_csv(self.df_path)

        # load the dataset to evaluate on
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        data_config = copy.deepcopy(config["dataset"])
        data_path = data_config[dataset_name.value].pop("path")
        self.dataset, self.torch_dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(data_path),
            batch_size=config.get("batch_size", 128),
            transform=self.transform,
            **data_config[dataset_name.value],
        )

        if config["bias_evaluation"]["train"]:  # True or
            # load the correct model to use as initialization
            self.model, self.model_out_dim = self.load_model(
                SSL_model=SSL_model,
            )
            logger.debug("model loaded")
            # check if the cache contains the embeddings already
            cache_file = (
                self.cache_path / f"{dataset_name.value}_{self.experiment_name}.pickle"
            )
            if cache_file.exists():
                print(f"Found cached file loading: {cache_file}")
                with open(cache_file, "rb") as file:
                    cached_dict = pickle.load(file)
                self.emb_space = cached_dict["embedding_space"]
                self.labels = cached_dict["labels"]
                del cached_dict
            else:
                self.emb_space, self.labels, _, _ = embed_dataset(
                    torch_dataset=self.torch_dataset,
                    model=self.model,
                    n_layers=n_layers,
                    memmap=False,
                    normalize=False,
                )
                del _
                # save the embeddings and issues to cache
                save_dict = {
                    "embedding_space": self.emb_space,
                    "labels": self.labels,
                }
                with open(cache_file, "wb") as handle:
                    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("initialized")

    @property
    @abstractmethod
    def experiment_name(self) -> str:
        pass

    @abstractmethod
    def split_dataframe_iterator(self) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
        pass

    def load_model(self, SSL_model: str):
        model, info, _ = Embedder.load_pretrained(
            SSL_model,
            return_info=True,
            n_head_layers=0,
        )
        # set the model in evaluation mode
        model = model.eval()
        # move to correct device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        return model, info.out_dim

    def evaluate(self):
        logger.debug("evaluation starting")
        # TODO fix
        if False:
            if self.df_path.exists() and not self.append_to_df:
                raise ValueError(
                    f"Dataframe already exists, remove to start: {self.df_path}"
                )
        self.dataset.return_path = False

        for e_type, config in self.eval_types:
            for (
                train_valid_range,
                test_range,
                split_name,
            ) in self.split_dataframe_iterator():
                if config["train"]:
                    if config.get("n_folds", None) is not None:
                        logger.debug("KFold starting")
                        k_fold = StratifiedGroupKFold(
                            n_splits=config["n_folds"],
                            random_state=self.seed,
                            shuffle=True,
                        )
                        labels = self.dataset.meta_data.loc[
                            train_valid_range, self.dataset.LBL_COL
                        ].values
                        groups = self.dataset.meta_data.loc[
                            train_valid_range, "subject_id"
                        ].values
                        fold_generator = k_fold.split(train_valid_range, labels, groups)
                        for i_fold, (train_range, valid_range) in tqdm(
                            enumerate(fold_generator),
                            total=config["n_folds"],
                            desc="K-Folds",
                        ):
                            self._run_evaluation_on_range(
                                e_type=e_type,
                                train_range=train_range,
                                eval_range=valid_range,
                                config=config,
                                add_run_info=f"Fold-{i_fold}",
                                split_name=split_name,
                                saved_model_path=None,
                            )
                if config["eval_test_performance"]:
                    self._run_evaluation_on_range(
                        e_type=e_type,
                        train_range=train_valid_range,
                        eval_range=test_range,
                        config=config,
                        add_run_info="Test",
                        split_name=split_name,
                        saved_model_path=self.model_path,
                        detailed_evaluation=True,
                    )

    def _run_evaluation_on_range(
        self,
        e_type: BaseEvalType,
        train_range: np.ndarray,
        eval_range: np.ndarray,
        config: dict,
        add_run_info: Optional[str] = None,
        split_name: Optional[str] = None,
        saved_model_path: Union[Path, str, None] = None,
        detailed_evaluation: bool = False,
    ):
        logger.debug("run evaluation on range starting")

        # W&B configurations
        if e_type in (EvalFineTuning, EvalBias) and self.log_wandb:
            _config = copy.deepcopy(self.config)
            if split_name is not None:
                _config["split_name"] = split_name
            wandb.init(
                config=_config,
                project=self.wandb_project_name,
            )
            wandb_run_name = f"{self.experiment_name}-{wandb.run.name}"
            if add_run_info is not None:
                wandb_run_name += f"-{add_run_info}"
            wandb.run.name = wandb_run_name
            wandb.run.save()
        if "train" in config and config["train"]:
            logger.debug("training starting")

            # get train / test set
            score_dict = e_type.evaluate(
                emb_space=self.emb_space,
                labels=self.labels,
                train_range=train_range,
                evaluation_range=eval_range,
                # only needed for fine-tuning
                dataset=self.dataset,
                model=self.model,
                model_out_dim=self.model_out_dim,
                log_wandb=self.log_wandb,
                saved_model_path=saved_model_path,
                # rest of the method specific parameters set with kwargs
                **config,
            )
        else:
            logger.debug("loading saved dict")
            # TODO make more dynamic if possible, configurable
            score_dict = self.load_and_find_row(e_type, "Test", "finetuning")

        if detailed_evaluation:
            print("°" * 20 + " labels " + "°" * 20)
            print(self.labels)
            # Detailed evaluation
            self.print_overall_result(e_type, score_dict)
            # Detailed evaluation per demographic
            eval_df = self.dataset.meta_data.iloc[eval_range].copy()
            logger.debug(f"eval_df copy: {eval_df}")

            eval_df.reset_index(drop=True, inplace=True)
            eval_df["targets"] = score_dict["targets"]
            eval_df["predictions"] = score_dict["predictions"]

            logger.debug(f"eval_df: {eval_df}")

            fst_types = self.print_fst_result(eval_df, score_dict)
            gender_types = self.print_gender_result(eval_df, score_dict)

            self.print_fst_gender_result(eval_df, fst_types, gender_types, score_dict)

            # Aggregate predictions per sample
            eval_df_aggregated = eval_df.groupby("subject_id").agg(
                {"targets": list, "predictions": list}
            )
            case_targets = (
                eval_df_aggregated["targets"]
                .apply(lambda x: max(set(x), key=x.count))
                .values
            )
            case_predictions = (
                eval_df_aggregated["predictions"]
                .apply(lambda x: max(set(x), key=x.count))
                .values
            )
            logger.debug(f"eval_df_aggregated: {eval_df_aggregated}")
            logger.debug(f"case_targets: {case_targets}")
            logger.debug(f"case_predictions: {case_predictions}")

            eval_df_aggregated["targets"] = case_targets
            eval_df_aggregated["predictions"] = case_predictions
            logger.debug(f"eval_df_aggregated - majority voted: {eval_df_aggregated}")

            print("*" * 20 + f" {e_type.name()} -> Case Agg. " + "*" * 20)
            self.print_eval_scores(
                y_true=case_targets,
                y_pred=case_predictions,
            )

            print("=" * 20 + " grouped output " + "~=" * 20)
            self.print_grouped_result(eval_df, group_by="sex")
            self.print_grouped_result(eval_df, group_by="fitzpatrick")

            print("=" * 20 + " grouped output per case using subgroup " + "~=" * 20)
            self.print_subgroup_results(eval_df, score_dict, group_by=["fitzpatrick"])
            self.print_subgroup_results(eval_df, score_dict, group_by=["sex"])
            # todo: add bins for age
            # self.print_subgroup_results(eval_df, score_dict, group_by=["age"])
            self.print_subgroup_results(
                eval_df, score_dict, group_by=["fitzpatrick", "sex"]
            )
            # self.print_subgroup_results(
            #    eval_df, score_dict, group_by=["fitzpatrick", "age"]
            # )
            # self.print_subgroup_results(eval_df, score_dict, group_by=["sex", "age"])
            # self.print_subgroup_results(
            #    eval_df, score_dict, group_by=["fitzpatrick", "sex", "age"]
            # )

            print("=" * 20 + " grouped output per case - majority voted" + "~=" * 20)
            self.print_grouped_result(eval_df_aggregated, group_by="sex")
            self.print_grouped_result(eval_df_aggregated, group_by="fitzpatrick")

        # finish the W&B run if needed
        if e_type in (EvalFineTuning, EvalBias) and self.log_wandb:
            wandb.finish()
        # TODO: fix
        if False:
            # save the results to the overall dataframe + save df
            self.df.loc[len(self.df)] = list(score_dict.values()) + [
                split_name,
                add_run_info,
                e_type.name(),
            ]
            self.df.to_csv(self.df_path, index=False)

    def print_subgroup_results(self, eval_df, score_dict, group_by: list[str]):
        def to_pascal_case(s: str) -> str:
            return "".join(word.capitalize() for word in s.split("_"))

        # Get all unique combinations of the group attributes
        grouped = eval_df.groupby(group_by)

        for group_values, _df in grouped:
            if isinstance(group_values, str):
                group_values = (group_values,)  # Make it always a tuple

            if _df.shape[0] == 0:
                continue  # Skip empty groups

            # Build header text
            group_info = ", ".join(
                f"{to_pascal_case(attr)}: {val}"
                for attr, val in zip(group_by, group_values)
            )

            print("~" * 20 + f" {group_info}, Support: {_df.shape[0]} " + "~" * 20)
            self.print_eval_scores(
                y_true=score_dict["targets"][_df.index.values],
                y_pred=score_dict["predictions"][_df.index.values],
            )

    def print_fst_gender_result(self, eval_df, fst_types, gender_types, score_dict):
        # Detailed evaluation per Fitzpatrick & Gender subgroup
        for fst in fst_types:
            for gender in gender_types:
                _df = eval_df[
                    (eval_df["fitzpatrick"] == fst) & (eval_df["sex"] == gender)
                ]
                if _df.shape[0] == 0:
                    continue  # Skip if no samples in subgroup
                print(
                    "~" * 20
                    + f" Fitzpatrick: {fst}, Gender: {gender}, Support: {_df.shape[0]} "
                    + "~" * 20
                )
                self.print_eval_scores(
                    y_true=score_dict["targets"][_df.index.values],
                    y_pred=score_dict["predictions"][_df.index.values],
                )

    def print_gender_result(self, eval_df, score_dict):
        gender_types = eval_df["sex"].unique()
        for gender in gender_types:
            _df = eval_df[eval_df["sex"] == gender]
            print("~" * 20 + f" Gender: {gender}, Support: {_df.shape[0]} " + "~" * 20)
            self.print_eval_scores(
                y_true=score_dict["targets"][_df.index.values],
                y_pred=score_dict["predictions"][_df.index.values],
            )
        return gender_types

    def print_fst_result(self, eval_df, score_dict):
        fst_types = eval_df["fitzpatrick"].unique()
        for fst in fst_types:
            _df = eval_df[eval_df["fitzpatrick"] == fst]
            print(
                "~" * 20 + f" Fitzpatrick: {fst}, Support: {_df.shape[0]} " + "~" * 20
            )
            self.print_eval_scores(
                y_true=score_dict["targets"][_df.index.values],
                y_pred=score_dict["predictions"][_df.index.values],
            )
        return fst_types

    def print_grouped_result(self, eval_df, group_by: str):
        group_types = eval_df[group_by].unique()
        group_name = group_by.capitalize()
        for group_value in group_types:
            _df = eval_df[eval_df[group_by] == group_value]
            print(
                "~" * 20
                + f" {group_name}: {group_value}, Support: {_df.shape[0]} "
                + "~" * 20
            )
            self.print_eval_scores(
                y_true=eval_df["targets"][_df.index.values],
                y_pred=eval_df["predictions"][_df.index.values],
            )
        return group_types

    def print_overall_result(self, e_type, score_dict):
        print("*" * 20 + f" {e_type.name()} " + "*" * 20)
        self.print_eval_scores(
            y_true=score_dict["targets"],
            y_pred=score_dict["predictions"],
        )

    def print_eval_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        if len(self.dataset.classes) == 2:
            f1 = f1_score(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=1,
                average="binary",
            )
            precision = precision_score(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=1,
                average="binary",
            )
            recall = recall_score(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=1,
                average="binary",
            )
            print(f"Bin. F1: {f1:.2f}")
            print(f"Bin. Precision: {precision:.2f}")
            print(f"Bin. Recall: {recall:.2f}")
        else:
            print(
                classification_report(
                    y_true=y_true,
                    y_pred=y_pred,
                    target_names=self.dataset.classes,
                )
            )
            b_acc = balanced_accuracy_score(
                y_true=y_true,
                y_pred=y_pred,
            )
            print(f"Balanced Acc: {b_acc}")

    def load_and_find_row(self, eval_type, add_run_info, split_name):
        if not self.df_path.exists():
            print(f"No results file found at: {self.df_path}")
            return None

        self.df = pd.read_csv(self.df_path)

        # Filter the dataframe to find the row matching the given parameters
        # todo: do not use magic texts
        filtered_row = self.df[
            (self.df["EvalType"] == "Standard_TRAIN_TEST")
            & (self.df["AdditionalRunInfo"] == add_run_info)
            & (self.df["SplitName"] == split_name)
        ]

        # todo fix where this is
        import ast

        def parse_targets(targets_str):
            # Replace spaces between numbers with commas
            targets_str = targets_str.replace(" ", ",")
            # Add square brackets to make it a valid list format if it's not already
            if not targets_str.startswith("["):
                targets_str = "[" + targets_str
            if not targets_str.endswith("]"):
                targets_str = targets_str + "]"
            # Now safely evaluate the string to a list using literal_eval
            result = [int(x) for x in ast.literal_eval(targets_str)]
            logger.debug(f"parse result: {result}")
            return pd.Series(result)

        # If we find a matching row, return the relevant information
        if not filtered_row.empty:
            result = filtered_row.iloc[-1]  # Get most recent added entry
            return {
                "score": result["Score"],
                "targets": parse_targets(result["EvalTargets"]),
                "predictions": parse_targets(result["EvalPredictions"]),
            }
        else:
            logger.debug(
                f'No matching row found for {"Standard_TRAIN_TEST"}, {add_run_info}, {split_name} in data: {self.df}'
            )
            return None
