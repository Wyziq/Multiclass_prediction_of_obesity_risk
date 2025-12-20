"""Registry of models used in the project.

Choose a model by name in `src/train.py`:
- logreg
- svm
- rf
- catboost (optional dependency)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_model(name: str):
    key = (name or "").strip().lower()

    if key in {"logreg", "logistic", "logistic_regression"}:
        return LogisticRegression(max_iter=3000)

    if key in {"svm", "svc"}:
        return SVC(kernel="rbf")

    if key in {"rf", "random_forest", "randomforest"}:
        return RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
        )

    if key in {"catboost", "cb"}:
        try:
            from catboost import CatBoostClassifier  # type: ignore
        except Exception as e:
            raise ImportError("CatBoost is not installed. Install it with: pip install catboost") from e

        return CatBoostClassifier(
            loss_function="MultiClass",
            verbose=False,
            random_seed=42,
        )

    raise ValueError(f"Unknown model name: {name}. Use: logreg | svm | rf | catboost")
