import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.trainers.eval_types.base import BaseEvalType


class EvalLin(BaseEvalType):
    @staticmethod
    def name() -> str:
        return "lin"

    @classmethod
    def evaluate(
        cls,
        emb_space: np.ndarray,
        labels: np.ndarray,
        train_range: np.ndarray,
        evaluation_range: np.ndarray,
        solver: str = "sag",
        tol: float = 0.1,
        max_iter: int = 100,
        **kwargs,
    ) -> dict:
        train, evaluation = cls.split_data(
            emb_space=emb_space,
            labels=labels,
            train_range=train_range,
            evaluation_range=evaluation_range,
        )
        X_train, y_train = train
        X_eval, y_eval = evaluation
        del train, evaluation

        lin = LogisticRegression(
            solver=solver,
            n_jobs=-1,
            tol=tol,
            max_iter=max_iter,
        )
        lin.fit(X_train, y_train)
        y_pred = lin.predict(X_eval)

        f1 = float(
            f1_score(
                y_eval,
                y_pred,
                average="macro",
            )
            * 100
        )
        return {
            "score": f1,
            "filenames": [],
            "indices": [i for i in range(len(y_eval))],
            "targets": y_eval,
            "predictions": y_pred,
        }
