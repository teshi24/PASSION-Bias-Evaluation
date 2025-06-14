from itertools import combinations
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np

from src.datasets.helper import DatasetName
from src.trainers.evaluation_trainer import EvaluationTrainer


class ExperimentCenterGeneralization(EvaluationTrainer):
    def __init__(
        self,
        dataset_name: DatasetName,
        config: dict,
        ckp_path: Optional[str] = None,
        SSL_model: str = "imagenet",
        output_path: Union[Path, str] = "assets/evaluation",
        cache_path: Union[Path, str] = "assets/evaluation/cache",
        model_path: Union[Path, str] = None,
        n_layers: int = 1,
        append_to_df: bool = False,
        wandb_project_name: str = "PASSION-Evaluation",
        log_wandb: bool = False,
    ):
        super().__init__(
            dataset_name=dataset_name,
            config=config,
            SSL_model=SSL_model,
            output_path=output_path,
            cache_path=cache_path,
            model_path=model_path,
            n_layers=n_layers,
            append_to_df=append_to_df,
            wandb_project_name=wandb_project_name,
            log_wandb=log_wandb,
        )

    @property
    def experiment_name(self) -> str:
        return "experiment_center"

    def split_dataframe_iterator(self) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
        test_range = self.dataset.meta_data[
            self.dataset.meta_data["Split"] == "TEST"
        ].index.values

        l_countries = set(self.dataset.meta_data["country"].unique())
        data_combinations = [
            {"train": list(x), "test": list(set(x) ^ l_countries)}
            for x in list(combinations(l_countries, 2))
        ]
        for split_dict in data_combinations:
            train_valid_range = self.dataset.meta_data[
                (self.dataset.meta_data["Split"] == "TRAIN")
                & (self.dataset.meta_data["country"].isin(split_dict["train"]))
            ].index.values
            train_valid_resampled_range = (
                self.dataset.meta_data[self.dataset.meta_data["Split"] == "TRAIN"]
                .sample(n=len(train_valid_range), random_state=self.seed)
                .index.values
            )

            split_name = f"TRAIN: {'_'.join(split_dict['train'])}, TEST: Standard"
            yield train_valid_range, test_range, split_name

            split_name = f"TRAIN: Standard (resampled: {'_'.join(split_dict['train'])}), TEST: Standard"
            yield train_valid_resampled_range, test_range, split_name
