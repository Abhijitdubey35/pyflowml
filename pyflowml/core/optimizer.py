"""
Optimizer — Dataset-size-aware model selection and strategy engine.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


class ModelOptimizer:
    """
    Intelligently selects candidate models based on dataset size,
    problem type, and available compute budget.

    Dataset Tiers:
    - small  (<10K rows):   All models
    - medium (10K–500K):    Skip SVM, KNN
    - large  (>500K rows):  Linear + Tree only
    """

    # Model pools per problem type
    _CLASSIFICATION_ALL = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
        "GradientBoosting": lambda: GradientBoostingClassifier(random_state=42),
        "SVM": lambda: SVC(probability=True, random_state=42, max_iter=2000),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=1),
    }

    _REGRESSION_ALL = {
        "Ridge": lambda: Ridge(),
        "Lasso": lambda: Lasso(max_iter=2000),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=42),
        "SVR": lambda: SVR(max_iter=2000),
        "KNN": lambda: KNeighborsRegressor(n_neighbors=5, n_jobs=1),
    }

    _SKIP_MEDIUM = {"SVM", "SVR", "KNN"}
    _SKIP_LARGE  = {"SVM", "SVR", "KNN", "GradientBoosting"}

    def __init__(self, problem_type: str, dataset_tier: str = "small"):
        self.problem_type = problem_type
        self.dataset_tier = dataset_tier

    def get_candidate_models(self) -> dict:
        """Return a dict of {name: model_instance} based on tier."""
        if self.problem_type == "classification":
            pool = dict(self._CLASSIFICATION_ALL)
        else:
            pool = dict(self._REGRESSION_ALL)

        # Add boosting if available
        if HAS_XGB:
            if self.problem_type == "classification":
                # No early_stopping_rounds / use_label_encoder: both are
                # version-fragile in the sklearn API (early stopping needs an
                # eval_set at fit time, and use_label_encoder was removed in
                # xgboost 2.0). Models train to n_estimators within the budget.
                pool["XGBoost"] = lambda: XGBClassifier(
                    n_estimators=200, eval_metric="logloss",
                    random_state=42, n_jobs=1
                )
            else:
                pool["XGBoost"] = lambda: XGBRegressor(
                    n_estimators=200, random_state=42, n_jobs=1
                )

        if HAS_LGBM:
            if self.problem_type == "classification":
                pool["LightGBM"] = lambda: LGBMClassifier(
                    n_estimators=200, random_state=42,
                    callbacks=None, n_jobs=1
                )
            else:
                pool["LightGBM"] = lambda: LGBMRegressor(
                    n_estimators=200, random_state=42, n_jobs=1
                )

        # Prune based on tier
        if self.dataset_tier == "medium":
            pool = {k: v for k, v in pool.items() if k not in self._SKIP_MEDIUM}
        elif self.dataset_tier == "large":
            pool = {k: v for k, v in pool.items() if k not in self._SKIP_LARGE}

        # Instantiate
        return {name: factory() for name, factory in pool.items()}

    @staticmethod
    def recommend_reason(model_name: str, profile: dict) -> str:
        """Generate a human-readable reason for why a model was selected as best."""
        reasons = {
            "RandomForest": "Handles non-linearity + robust to outliers + no scaling needed",
            "LightGBM": "Fast gradient boosting + handles large datasets efficiently",
            "XGBoost": "High accuracy gradient boosting with regularization",
            "LogisticRegression": "Fast, interpretable, and great for linearly separable data",
            "Ridge": "Robust linear regression with L2 regularization",
            "GradientBoosting": "Strong ensemble method with high predictive power",
            "SVM": "Effective in high-dimensional spaces",
            "KNN": "Non-parametric, captures local patterns",
            "Lasso": "Linear regression with built-in feature selection",
        }
        return reasons.get(model_name, "Best performance on validation set")
