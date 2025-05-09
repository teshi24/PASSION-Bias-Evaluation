import copy
import pickle
import re
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torchvision import transforms
from tqdm import tqdm
from wandb.sdk.lib.file_stream_utils import split_files

from src.datasets.helper import DatasetName, get_dataset
from src.models.embedder import Embedder
from src.models.helper import embed_dataset
from src.trainers.eval_types.base import BaseEvalType
from src.trainers.eval_types.dummy_classifier import (
    EvalDummyConstant,
    EvalDummyMostFrequent,
    EvalDummyUniform,
)
from src.trainers.eval_types.fine_tuning import EvalFineTuning
from src.utils.evaluator import BiasEvaluator
from src.utils.utils import fix_random_seeds

eval_type_dict = {
    # Baselines
    "dummy_most_frequent": EvalDummyMostFrequent,
    "dummy_uniform": EvalDummyUniform,
    "dummy_constant": EvalDummyConstant,
    # Models
    "fine_tuning": EvalFineTuning,
}


class EvaluationTrainer(ABC, object):
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

        self.df_description = f"{self.experiment_name}_{self.dataset_name.value}"
        self.df_name = f"{self.df_description}.csv"
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
                "FileNames",
                "Indices",
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
        self.input_size = config["input_size"]
        self.transform = transforms.Compose(
            [
                # transforms.Resize((144, 144)), # todo: activate for input_size 128
                transforms.Resize((256, 256)),  # activate for input_size 224
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        data_config = copy.deepcopy(config["dataset"])[dataset_name.value]
        data_path = data_config.pop("path")
        self.evaluator = BiasEvaluator(
            passion_exp=self.df_description,
            eval_data_path=self.output_path,
            dataset_dir=Path(data_path),
            meta_data_file=data_config["meta_data_file"],
            split_file=data_config["split_file"],
        )

        self.dataset, self.torch_dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(data_path),
            batch_size=config.get("batch_size", 128),
            transform=self.transform,
            # num_workers=config.get("num_workers", 4),
            **data_config,
        )

        # load the correct model to use as initialization
        if SSL_model == "GoogleDermFound":
            self.dataset.return_embedding = True
            self.torch_dataset.dataset.return_embedding = True
        else:
            self.model, self.model_out_dim = self.load_model(SSL_model=SSL_model)
        #
        # check if the cache contains the embeddings already
        logger.debug("embed data")
        cache_file = (
            self.cache_path / f"{dataset_name.value}_{self.experiment_name}.pickle"
        )
        if cache_file.exists():
            print(f"Found cached file loading: {cache_file}")
            with open(cache_file, "rb") as file:
                cached_dict = pickle.load(file)
            self.emb_space = cached_dict["embedding_space"]
            self.labels = cached_dict["labels"]
            # todo: figure out where this is reloaded / should be used
            self.paths = cached_dict["paths"]
            self.indices = cached_dict["indices"]
            del cached_dict
        else:
            (
                self.emb_space,
                self.labels,
                self.images,
                self.paths,
                self.indices,
            ) = embed_dataset(
                torch_dataset=self.torch_dataset,
                model=self.model,
                n_layers=n_layers,
                # memmap=False,
                normalize=False,
            )
            # todo: what with those imgs
            # save the embeddings and issues to cache
            save_dict = {
                "embedding_space": self.emb_space,
                "labels": self.labels,
                "paths": self.paths,
                "indices": self.indices,
            }
            with open(cache_file, "wb") as handle:
                pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    @abstractmethod
    def experiment_name(self) -> str:
        pass

    @abstractmethod
    def split_dataframe_iterator(self) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
        pass

    def load_model(self, SSL_model: str):
        logger.debug("load model")
        model, info, _ = Embedder.load_pretrained(
            SSL_model,
            return_info=True,
            n_head_layers=0,
        )
        # set the model in evaluation mode
        model = model.eval()
        # move to correct device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"device: {device}")
        model = model.to(device)
        return model, info.out_dim

    def evaluate(self):
        if self.df_path.exists() and not self.append_to_df:
            raise ValueError(
                f"Dataframe already exists, remove to start: {self.df_path}"
            )

        for e_type, config in self.eval_types:
            for (
                train_valid_range,
                test_range,
                split_name,
            ) in self.split_dataframe_iterator():
                if (
                    config.get("train_from_scratch", True)
                    and config.get("n_folds", None) is not None
                ):
                    k_fold = StratifiedGroupKFold(
                        n_splits=config["n_folds"],
                        random_state=self.seed,
                        shuffle=True,
                    )
                    # todo: here, maybe add the minority groups if needed, e.g. skin type
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
        self.configure_wandb(add_run_info, e_type, split_name)
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
        # save the results to the overall dataframe + save df
        self.df.loc[len(self.df)] = list(score_dict.values()) + [
            split_name,
            add_run_info,
            e_type.name(),
        ]
        self.df.to_csv(self.df_path, index=False)
        if detailed_evaluation:
            # Detailed evaluation
            print("*" * 20 + f" {e_type.name()} " + "*" * 20)
            self.print_eval_scores(
                y_true=score_dict["targets"],
                y_pred=score_dict["predictions"],
            )

            run_detailed_evaluation = config.get("detailed_evaluation", False)
            if run_detailed_evaluation:
                print("=" * 20 + f" {e_type.name()} = my analysis " + "=" * 20)
                self.print_eval_scores_bias_old(score_dict)

                self.evaluator.run_full_evaluation(
                    self.df.iloc[[-1]],
                    add_run_info=add_run_info,
                    run_detailed_evaluation=run_detailed_evaluation,
                )
            else:
                self.evaluator.run_evaluation_without_metadata(
                    self.df.iloc[[-1]], add_run_info=add_run_info
                )

        self.finish_wandb(e_type)
        # save the results to the overall dataframe + save df
        # self.df.loc[len(self.df)] = list(score_dict.values()) + [
        #     split_name,
        #     add_run_info,
        #     e_type.name(),
        # ]
        # self.df.to_csv(self.df_path, index=False)

    def finish_wandb(self, e_type):
        # finish the W&B run if needed
        if e_type is EvalFineTuning and self.log_wandb:
            wandb.finish()

    def configure_wandb(self, add_run_info, e_type, split_name):
        # W&B configurations
        if e_type is EvalFineTuning and self.log_wandb:
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
            wandb.log({"log": "Your message"})
            print({"Wandb initialized", wandb.run})

    def create_results(self, df_results):
        df_labels = pd.read_csv("data/PASSION/label.csv")
        df_split = pd.read_csv("data/PASSION/PASSION_split.csv")

        def extract_subject_id(path: str):
            pattern = r"([A-Za-z]+[0-9]+)"
            match = re.search(pattern, path)
            if match:
                return str(match.group(1)).strip()
            else:
                return np.nan

        # # Parse the image_paths and arrays
        # def parse_image_paths(s):
        #     s = s.replace("\n", " ").replace("'", '"')
        #     s = s.replace("[", "").replace("]", "")
        #     return s.split()
        #
        # def parse_numpy_array(s):
        #     return np.fromstring(s.strip("[]").replace("\n", " "), sep=" ", dtype=int)
        #
        # df_results["filenames"] = df_results["filenames"].apply(parse_image_paths)
        # df_results["indices"] = df_results["indices"].apply(parse_numpy_array)
        # df_results["targets"] = df_results["targets"].apply(parse_numpy_array)
        # df_results["predictions"] = df_results["predictions"].apply(
        #     parse_numpy_array
        # )

        # Flatten the DataFrame into one row per image path
        rows = []
        unique_subject_ids = []
        for img_name, idx, lbl, pred in zip(
            df_results["filenames"],
            df_results["indices"],
            df_results["targets"],
            df_results["predictions"],
        ):
            subject_id = extract_subject_id(img_name)
            labels = df_labels[df_labels["subject_id"] == subject_id].iloc[0]
            split = df_split[df_split["subject_id"] == subject_id].iloc[0]
            rows.append(
                {
                    "correct": lbl == pred,
                    "image_path": img_name,
                    "index": idx,
                    "targets": lbl,
                    "predictions": pred,
                    **labels.to_dict(),
                    **split.drop("subject_id").to_dict(),
                }
            )
            if subject_id not in unique_subject_ids:
                unique_subject_ids.append(subject_id)

        print(len(unique_subject_ids))

        return pd.DataFrame(rows)

    def print_eval_scores_bias_old(self, df_results):
        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! old analysis !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )
        logger.debug(df_results)

        df_calc_in = self.create_results(df_results)
        labels = [0, 1, 2, 3]
        target_names = ["Eczema", "Fungal", "Others", "Scabies"]
        print_details = True

        def print_detailed_bias_eval_scores(y_true: np.ndarray, y_pred: np.ndarray):
            if print_details:
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                print("Confusion Matrix:")
                print(cm)

                def get_tp_fp_fn_tn(cm, class_index):
                    total_classifications = cm.sum()
                    tp = cm[class_index, class_index]
                    fp = cm[:, class_index].sum() - tp
                    fn = cm[class_index, :].sum() - tp
                    tn = total_classifications - (tp + fp + fn)
                    return tp, fp, fn, tn

                for i, name in enumerate(target_names):
                    tp, fp, fn, tn = get_tp_fp_fn_tn(cm, i)
                    print(f"{name} — TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

            print(
                classification_report(
                    y_true=y_true,
                    y_pred=y_pred,
                    labels=labels,
                    target_names=target_names,
                    zero_division=0,
                )
            )
            b_acc = balanced_accuracy_score(
                y_true=y_true,
                y_pred=y_pred,
            )
            print(f"Balanced Acc: {b_acc}")

        def print_grouped_result(eval_df, group_by: str):
            group_types = sorted(eval_df[group_by].unique())
            group_name = group_by.capitalize()
            for group_value in group_types:
                _df = eval_df[eval_df[group_by] == group_value]
                print(
                    "~" * 20
                    + f" {group_name}: {group_value}, Support: {_df.shape[0]} "
                    + "~" * 20
                )
                print_detailed_bias_eval_scores(
                    y_true=eval_df["targets"][_df.index.values],
                    y_pred=eval_df["predictions"][_df.index.values],
                )
            return group_types

        def print_subgroup_results(eval_df, group_by: list[str]):
            def to_pascal_case(s: str) -> str:
                return "".join(word.capitalize() for word in s.split("_"))

            # Get all unique combinations of the group attributes
            grouped = sorted(eval_df.groupby(group_by))

            for group_values, _df in grouped:
                if isinstance(group_values, str):
                    group_values = (group_values,)  # Make it always a tuple

                if _df.shape[0] == 0:
                    print(
                        f"no support: group_by: {group_by}, group_values: {group_values}"
                    )
                    continue  # Skip empty groups

                # Build header text
                group_info = ", ".join(
                    f"{to_pascal_case(attr)}: {val}"
                    for attr, val in zip(group_by, group_values)
                )

                print("~" * 20 + f" {group_info}, Support: {_df.shape[0]} " + "~" * 20)
                print_detailed_bias_eval_scores(
                    y_true=eval_df["targets"][_df.index.values],
                    y_pred=eval_df["predictions"][_df.index.values],
                )

        def do_calculations(data):
            # Detailed evaluation
            print("*" * 20 + f" overall " + "*" * 20)
            print_detailed_bias_eval_scores(
                y_true=data["targets"],
                y_pred=data["predictions"],
            )

            print("=" * 20 + " now more dynamic (grouped) " + "=" * 20)
            print_grouped_result(data, group_by="fitzpatrick")
            print_grouped_result(data, group_by="sex")

            print("=" * 20 + " grouped output per case using subgroup " + "=" * 20)
            print("dataset")
            bins = list(range(0, 100, 5))
            labels = [f"{i}-{i + 4}" for i in bins[:-1]]
            data["ageGroup"] = pd.cut(
                data["age"], bins=bins, labels=labels, right=False
            )
            print(data)

            print_subgroup_results(data, group_by=["fitzpatrick"])
            print_subgroup_results(data, group_by=["sex"])
            print_subgroup_results(data, group_by=["ageGroup"])
            print_subgroup_results(data, group_by=["fitzpatrick", "sex"])
            print_subgroup_results(data, group_by=["fitzpatrick", "ageGroup"])
            print_subgroup_results(data, group_by=["sex", "ageGroup"])
            print_subgroup_results(data, group_by=["fitzpatrick", "sex", "ageGroup"])

        do_calculations(df_calc_in)

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
            try:
                print(
                    classification_report(
                        y_true=y_true,
                        y_pred=y_pred,
                        target_names=self.dataset.classes,
                    )
                )
            except Exception as e:
                print(f"Error generating classification report: {e}")
            b_acc = balanced_accuracy_score(
                y_true=y_true,
                y_pred=y_pred,
            )
            print(f"Balanced Acc: {b_acc}")
