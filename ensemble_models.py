from __future__ import annotations

import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=250,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=data.random_seed,
                    n_jobs=1,
                ),
            ),
        ]
    )

    model.fit(data.train_features, data.train_target)

    probabilities_all = model.predict_proba(data.test_features)

    classes = model.named_steps["classifier"].classes_
    positive_index = int(np.flatnonzero(classes == 1)[0])

    probabilities = probabilities_all[:, positive_index]
    predictions = (probabilities >= 0.5).astype(int)

    y_test = np.asarray(data.test_target).ravel()

    return EnsembleResult(
        model_name="Extra Trees (250 trees, min_samples_leaf=3)",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        recall=float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        f1=float(
            f1_score(y_test, predictions, zero_division=0)
        ),
    )