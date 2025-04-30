import copy
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
import wandb
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_lr_finder import LRFinder
from torchinfo import summary
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from src.models.classifiers import LinearClassifier
from src.optimizers.utils import get_optimizer_type
from src.trainers.eval_types.base import BaseEvalType
from src.utils.utils import (
    EarlyStopping,
    restart_from_checkpoint,
    save_checkpoint,
    set_requires_grad,
)


class EvalFineTuning(BaseEvalType):
    @classmethod
    def train_transform(cls):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(cls.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                # Experiments showed color jitter hinders performance,
                # but check again if problems with models and datasets arise
                # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    @classmethod
    def val_transform(cls):
        resize_size = 256 if cls.input_size == 224 else 160
        return transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(cls.input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    @staticmethod
    def name() -> str:
        return "finetuning"

    @classmethod
    def evaluate(
        cls,
        train_range: np.ndarray,
        evaluation_range: np.ndarray,
        dataset: torch.utils.data.Dataset,
        model: Optional[torch.nn.Module],
        model_out_dim: int,
        learning_rate: float,
        batch_size: int,
        input_size: int,
        train_epochs: int,
        warmup_epochs: int,
        early_stopping_patience: int,
        use_bn_in_head: bool,
        dropout_in_head: float,
        num_workers: int,
        saved_model_path: Union[Path, str, None] = None,
        find_optimal_lr: bool = False,
        use_lr_scheduler: bool = False,
        log_wandb: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> dict:
        cls.input_size = input_size
        classifier, model = cls.create_classifier(
            dataset, dropout_in_head, model, model_out_dim, use_bn_in_head
        )
        # get dataloader for batched compute
        train_loader, eval_loader = cls.get_train_eval_loaders(
            dataset=dataset,
            train_range=train_range,
            evaluation_range=evaluation_range,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        device = cls.get_device(model)
        classifier.to(device)

        cls.configure_classifier_base(classifier, debug, log_wandb, model, train_loader)
        # loss function, optimizer, scores
        criterion = torch.nn.CrossEntropyLoss(
            weight=train_loader.dataset.get_class_weights(),
        )
        criterion = criterion.to(device)
        optimizer = cls.configure_optimizer(
            classifier,
            criterion,
            device,
            find_optimal_lr,
            learning_rate,
            log_wandb,
            train_loader,
        )

        # we use early stopping to speed up the training
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            log_messages=debug,
        )

        if use_lr_scheduler:
            # define the learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=train_epochs,
                eta_min=0,
            )

        # load the model from checkpoint if provided
        to_restore = {"epoch": 0}
        # TODO: fix this here
        if False:
            if saved_model_path is not None:
                restart_from_checkpoint(
                    Path(saved_model_path) / "checkpoints" / "model_best.pth",
                    run_variables=to_restore,
                    classifier=classifier,
                    optimizer=optimizer,
                    loss=criterion,
                )
        start_epoch = to_restore["epoch"]

        # define metrics
        metric_param = {
            "task": "multiclass",
            "num_classes": classifier.fc.num_labels,
            "average": "macro",
        }
        loss_metric_train = torchmetrics.MeanMetric().to(device)
        f1_score_train = torchmetrics.F1Score(**metric_param).to(device)

        loss_metric_val = torchmetrics.MeanMetric().to(device)
        f1_score_val = torchmetrics.F1Score(**metric_param).to(device)
        precision_val = torchmetrics.Precision(**metric_param).to(device)
        recall_val = torchmetrics.Recall(**metric_param).to(device)
        auroc_val = torchmetrics.AUROC(
            task=metric_param["task"],
            num_classes=metric_param["num_classes"],
        ).to(device)

        # start training
        epoch, step = start_epoch, 0
        eval_scores_dict = {
            "f1": {
                "metric": f1_score_val,
                "scores": [],
            },
            "precision": {
                "metric": precision_val,
                "scores": [],
            },
            "recall": {
                "metric": recall_val,
                "scores": [],
            },
            "auroc": {
                "metric": auroc_val,
                "scores": [],
            },
        }
        l_loss_val = []
        best_val_score = 0
        best_model_wts = copy.deepcopy(classifier.state_dict())
        for epoch in tqdm(
            range(epoch, train_epochs),
            total=train_epochs,
            desc="Model Training",
        ):
            cls.freeze_as_needed(classifier, epoch, warmup_epochs)

            # training
            classifier.train()
            for img, target in train_loader:
                img = img.to(device)
                target = target.to(device)

                optimizer.zero_grad()

                pred = classifier(img)
                loss = criterion(pred, target)

                loss.backward()
                optimizer.step()
                if use_lr_scheduler:
                    scheduler.step()

                # W&B logging if needed
                if log_wandb:
                    log_dict = {
                        "train_loss": loss.item(),
                        "train_f1": f1_score_train(pred, target),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "weight_decay": optimizer.param_groups[0]["weight_decay"],
                        "epoch": epoch,
                        "step": step,
                    }
                    wandb.log(log_dict)
                # add to overall metrics
                loss_metric_train.update(loss.detach())
                f1_score_train.update(pred, target)
                step += 1

            # Evaluation
            classifier.eval()
            for img, _, target, _ in eval_loader:
                img = img.to(device)
                target = target.to(device)
                with torch.no_grad():
                    pred = classifier(img)
                    loss = criterion(pred, target)
                loss_metric_val.update(loss)
                for _score_dict in eval_scores_dict.values():
                    _score_dict["metric"].update(pred, target)
            l_loss_val.append(loss_metric_val.compute())
            for _score_dict in eval_scores_dict.values():
                _score_dict["scores"].append(_score_dict["metric"].compute())
            # check if we have new best model
            if eval_scores_dict["f1"]["scores"][-1] > best_val_score:
                best_val_score = eval_scores_dict["f1"]["scores"][-1]
                best_model_wts = copy.deepcopy(classifier.state_dict())
            # check early stopping
            early_stopping(l_loss_val[-1])
            if early_stopping.early_stop:
                if debug:
                    print("EarlyStopping, evaluation did not decrease.")
                break
            # W&B logging if needed
            if log_wandb:
                log_dict = {
                    "eval_loss": l_loss_val[-1],
                    "epoch": epoch,
                    "step": step,
                }
                for score_name, _score_dict in eval_scores_dict.items():
                    log_dict[f"eval_{score_name}"] = _score_dict["scores"][-1]
                wandb.log(log_dict)

        # get the best epoch in terms of F1 score
        wandb.unwatch()
        best_epoch = cls.get_best_epoch(
            epoch, eval_scores_dict, l_loss_val, log_wandb, step
        )
        classifier.load_state_dict(best_model_wts)
        if saved_model_path is not None:
            cls.save_model_checkpoint(
                classifier, criterion, epoch, optimizer, saved_model_path
            )

        # create eval predictions for saving
        img_names, targets, predictions, indices = [], [], [], []
        classifier.eval()
        for img, img_name, target, index in eval_loader:
            img = img.to(device)
            target = target.to(device)
            with torch.no_grad():
                pred = classifier(img)
            targets.append(target.cpu())
            predictions.append(pred.cpu())

            img_names.append(img_name)
            indices.append(index)
        img_names = torch.concat(img_names).numpy()
        targets = torch.concat(targets).cpu().numpy()
        predictions = torch.concat(predictions).argmax(dim=-1).cpu().numpy()
        indices = torch.concat(indices).numpy()
        results = {
            "score": float(eval_scores_dict["f1"]["scores"][best_epoch] * 100),
            "filenames": img_names,
            "indices": indices,
            "targets": targets,
            "predictions": predictions,
        }
        logger.debug(f"evaluation results: {results}")
        return results

    @classmethod
    def save_model_checkpoint(
        cls, classifier, criterion, epoch, optimizer, saved_model_path
    ):
        save_dict = {
            "arch": type(classifier).__name__,
            "epoch": epoch,
            "classifier": classifier,
            "optimizer": optimizer.state_dict(),
            "loss": criterion.state_dict(),
        }
        save_checkpoint(
            run_dir=saved_model_path,
            save_dict=save_dict,
            epoch=epoch,
            save_best=True,
        )

    @classmethod
    def get_best_epoch(cls, epoch, eval_scores_dict, l_loss_val, log_wandb, step):
        best_epoch = torch.Tensor(eval_scores_dict["f1"]["scores"]).argmax()
        if log_wandb:
            log_dict = {
                "best_eval_epoch": best_epoch,
                "best_eval_loss": l_loss_val[best_epoch],
                "epoch": epoch,
                "step": step,
            }
            for score_name, _score_dict in eval_scores_dict.items():
                log_dict[f"best_eval_{score_name}"] = _score_dict["scores"][best_epoch]
            wandb.log(log_dict)
        return best_epoch

    @classmethod
    def freeze_as_needed(cls, classifier, epoch, warmup_epochs):
        if epoch >= warmup_epochs:
            # make sure the classifier and backbone get trained
            set_requires_grad(classifier, True)
        else:
            # freeze the backbone and let only the classifier be trained
            set_requires_grad(classifier, True)
            if hasattr(classifier, "backbone"):
                set_requires_grad(classifier.backbone, False)

    @classmethod
    def configure_optimizer(
        cls,
        classifier,
        criterion,
        device,
        find_optimal_lr,
        learning_rate,
        log_wandb,
        train_loader,
    ):
        optimizer_cls = get_optimizer_type(optimizer_name="adam")
        optimizer = optimizer_cls(
            params=classifier.parameters(),
            lr=learning_rate,
        )
        if find_optimal_lr:
            # automatic learning rate finder
            lr_finder = LRFinder(classifier, optimizer, criterion, device=device)
            lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
            lrs = lr_finder.history["lr"]
            losses = lr_finder.history["loss"]
            # log the LRFinder plot
            fig, ax = plt.subplots()
            lr_finder.plot(ax=ax)
            if log_wandb:
                wandb.log({"LRFinder_Plot": wandb.Image(fig)})
            # to reset the model and optimizer to their initial state
            lr_finder.reset()
            try:
                min_grad_idx = (np.gradient(np.array(losses))).argmin()
                best_lr = lrs[min_grad_idx]
                optimizer = optimizer_cls(
                    params=classifier.parameters(),
                    lr=best_lr,
                )
            except ValueError:
                print("Failed to compute the gradients. Relying on default lr.")
        return optimizer

    @classmethod
    def configure_classifier_base(
        cls, classifier, debug, log_wandb, model, train_loader
    ):
        # make sure the classifier can get trained
        set_requires_grad(classifier, True)
        if debug and model is not None:
            try:
                summary(classifier, input_size=(1, 3, cls.input_size, cls.input_size))
            except RuntimeError:
                print("Summary can not be displayed for a Huggingface model.")
                print(
                    f"Number of parameters backbone: {classifier.backbone.model.num_parameters():,}"
                )
        if log_wandb:
            wandb.watch(classifier, log="all", log_freq=len(train_loader))

    @classmethod
    def get_device(cls, model):
        if model is not None:
            device = model.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"device: {device}")
        return device

    @classmethod
    def create_classifier(
        cls, dataset, dropout_in_head, model, model_out_dim, use_bn_in_head
    ):
        # create the classifier
        classifier_list = []
        if model is not None:
            model = copy.deepcopy(model)
            classifier_list = [
                ("backbone", model),
                ("flatten", torch.nn.Flatten()),
            ]
        classifier_list.append(
            (
                "fc",
                LinearClassifier(
                    model_out_dim,
                    dataset.n_classes,
                    large_head=False,
                    use_bn=use_bn_in_head,
                    dropout_rate=dropout_in_head,
                ),
            ),
        )
        classifier = torch.nn.Sequential(OrderedDict(classifier_list))
        return classifier, model

    @classmethod
    def get_train_eval_loaders(
        cls,
        dataset: torch.utils.data.Dataset,
        train_range: np.ndarray,
        evaluation_range: np.ndarray,
        batch_size: int,
        num_workers: int,
    ):
        train_dataset = copy.deepcopy(dataset)
        train_dataset.transform = cls.train_transform()
        train_dataset.train_data_only = True
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_range),
            num_workers=num_workers,
            drop_last=True,
            shuffle=False,
        )
        del train_dataset

        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.transform = cls.val_transform()
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(evaluation_range),
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
        )
        del eval_dataset
        return train_loader, eval_loader
