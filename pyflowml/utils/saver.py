"""
ModelSaver — Versioned model saving and loading with metadata.
"""

import os
import pickle
import json
from datetime import datetime
import joblib
from pyflowml.monitoring.logger import get_logger

logger = get_logger("ModelSaver")

DEFAULT_DIR = "saved_models"


class ModelSaver:
    """
    Save and load models with versioning and metadata.

    Example
    -------
    >>> ModelSaver.save(clf, "titanic_model", metadata={"accuracy": 0.92})
    >>> clf = ModelSaver.load("saved_models/titanic_model_v1_20260415.pkl")
    """

    @staticmethod
    def save(model, name: str, directory: str = DEFAULT_DIR,
             metadata: dict = None, pipeline=None, use_joblib: bool = True) -> str:
        """
        Save a model with versioned filename and optional metadata.

        Parameters
        ----------
        model     : Trained model object (e.g. AutoClassifier/AutoRegressor)
        name      : Base name for the model file
        directory : Output directory
        metadata  : Dict of extra info (metrics, features, etc.)
        pipeline  : Optional fitted preprocessing pipeline. When given, it is
                    bundled with the model so the artifact can predict directly
                    from raw data (use ``ModelSaver.predict``).
        use_joblib: Use joblib (True) or pickle (False)

        Returns
        -------
        str : Full path to saved model file
        """
        os.makedirs(directory, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{date_str}.pkl"
        filepath = os.path.join(directory, filename)

        # Persist the FULL wrapper, not just best_model_, so the fitted label
        # encoder travels with it (otherwise reloaded classifiers return raw
        # encoded integers and can't reproduce predictions). When a fitted
        # pipeline is supplied, bundle both into one self-contained artifact.
        if pipeline is not None:
            payload = {"__pyflow_bundle__": True, "pipeline": pipeline, "model": model}
        else:
            payload = model

        if use_joblib:
            joblib.dump(payload, filepath)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(payload, f)

        # Save metadata
        inner = getattr(model, "best_model_", model)
        if metadata is not None:
            meta = {
                "name": name,
                "saved_at": date_str,
                "model_type": type(inner).__name__,
                "bundled_pipeline": pipeline is not None,
                **metadata,
            }
            meta_path = filepath.replace(".pkl", "_meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)
            logger.info(f"Metadata saved: {meta_path}")

        logger.info(f"Model saved: {filepath}")
        return filepath

    @staticmethod
    def load(filepath: str, use_joblib: bool = True):
        """
        Load a previously saved model.

        Parameters
        ----------
        filepath  : Path to the .pkl model file
        use_joblib: Whether file was saved with joblib (default True)

        Returns
        -------
        Loaded model object
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")

        if use_joblib:
            model = joblib.load(filepath)
        else:
            with open(filepath, "rb") as f:
                model = pickle.load(f)

        logger.info(f"Model loaded: {filepath}")

        # Load and display metadata if it exists
        meta_path = filepath.replace(".pkl", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            logger.info(f"Metadata: {meta}")

        return model

    @staticmethod
    def predict(payload, X):
        """
        Predict from a loaded payload, whether it is a bare model wrapper or a
        bundle saved with a preprocessing pipeline.

        Example
        -------
        >>> bundle = ModelSaver.load("saved_models/my_model_20260615.pkl")
        >>> preds = ModelSaver.predict(bundle, raw_df)
        """
        if isinstance(payload, dict) and payload.get("__pyflow_bundle__"):
            X = payload["pipeline"].transform(X)
            return payload["model"].predict(X)
        return payload.predict(X)

    @staticmethod
    def list_saved(directory: str = DEFAULT_DIR) -> list:
        """List all saved models in a directory."""
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return []
        files = [f for f in os.listdir(directory) if f.endswith(".pkl")]
        if not files:
            print("No saved models found.")
        else:
            print(f"\n  Saved models in '{directory}':")
            for f in sorted(files):
                path = os.path.join(directory, f)
                size_kb = os.path.getsize(path) / 1024
                print(f"  • {f}  ({size_kb:.1f} KB)")
        return files
