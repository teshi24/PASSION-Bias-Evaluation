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
from src.trainers.eval_types.dummy_classifier import (
    EvalDummyConstant,
    EvalDummyMostFrequent,
    EvalDummyUniform,
)
from src.trainers.eval_types.fine_tuning import EvalFineTuning
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
            num_workers=config.get("num_workers", 4),
            **data_config[dataset_name.value],
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
            self.emb_space, self.labels, self.paths, self.indices = embed_dataset(
                torch_dataset=self.torch_dataset,
                model=self.model,
                n_layers=n_layers,
                # memmap=False,
                normalize=False,
            )
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
                if config.get("n_folds", None) is not None:
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
        if detailed_evaluation:
            # Detailed evaluation
            print("*" * 20 + f" {e_type.name()} " + "*" * 20)
            self.print_eval_scores(
                y_true=score_dict["targets"],
                y_pred=score_dict["predictions"],
            )
            # Detailed evaluation per demographic
            eval_df = self.dataset.meta_data.iloc[eval_range].copy()
            eval_df.reset_index(drop=True, inplace=True)
            eval_df["targets"] = score_dict["targets"]
            eval_df["predictions"] = score_dict["predictions"]
            fst_types = eval_df["fitzpatrick"].unique()
            for fst in fst_types:
                _df = eval_df[eval_df["fitzpatrick"] == fst]
                print(
                    "~" * 20
                    + f" Fitzpatrick: {fst}, Support: {_df.shape[0]} "
                    + "~" * 20
                )
                self.print_eval_scores(
                    y_true=score_dict["targets"][_df.index.values],
                    y_pred=score_dict["predictions"][_df.index.values],
                )
            gender_types = eval_df["sex"].unique()
            for gender in gender_types:
                _df = eval_df[eval_df["sex"] == gender]
                print(
                    "~" * 20 + f" Gender: {gender}, Support: {_df.shape[0]} " + "~" * 20
                )
                self.print_eval_scores(
                    y_true=score_dict["targets"][_df.index.values],
                    y_pred=score_dict["predictions"][_df.index.values],
                )
            # Aggregate predictions per sample
            eval_df = eval_df.groupby("subject_id").agg(
                {"targets": list, "predictions": list}
            )
            case_targets = (
                eval_df["targets"].apply(lambda x: max(set(x), key=x.count)).values
            )
            case_predictions = (
                eval_df["predictions"].apply(lambda x: max(set(x), key=x.count)).values
            )
            print("*" * 20 + f" {e_type.name()} -> Case Agg. " + "*" * 20)
            self.print_eval_scores(
                y_true=case_targets,
                y_pred=case_predictions,
            )
        self.finish_wandb(e_type)
        # save the results to the overall dataframe + save df
        self.df.loc[len(self.df)] = list(score_dict.values()) + [
            split_name,
            add_run_info,
            e_type.name(),
        ]
        self.df.to_csv(self.df_path, index=False)

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
