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
             metadata: dict = None, use_joblib: bool = True) -> str:
        """
        Save a model with versioned filename and optional metadata.

        Parameters
        ----------
        model     : Trained model object
        name      : Base name for the model file
        directory : Output directory
        metadata  : Dict of extra info (metrics, features, etc.)
        use_joblib: Use joblib (True) or pickle (False)

        Returns
        -------
        str : Full path to saved model file
        """
        os.makedirs(directory, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{date_str}.pkl"
        filepath = os.path.join(directory, filename)

        # Save model
        inner = getattr(model, "best_model_", model)
        if use_joblib:
            joblib.dump(inner, filepath)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(inner, f)

        # Save metadata
        if metadata is not None:
            meta = {
                "name": name,
                "saved_at": date_str,
                "model_type": type(inner).__name__,
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
