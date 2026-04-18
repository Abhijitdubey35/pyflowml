"""
Core Engine — Orchestrates the full PyFlowML pipeline.
"""

import time
from pyflowml.core.profiler import DataProfiler
from pyflowml.core.optimizer import ModelOptimizer
from pyflowml.data.cleaner import DataCleaner
from pyflowml.data.memory import MemoryOptimizer
from pyflowml.data.splitter import DataSplitter
from pyflowml.preprocessing.pipeline import SmartPipeline
from pyflowml.models.auto import AutoClassifier, AutoRegressor
from pyflowml.evaluation.reporter import Reporter
from pyflowml.monitoring.tracker import StepTracker
from pyflowml.monitoring.logger import get_logger

logger = get_logger("PyFlowEngine")


class PyFlowEngine:
    """
    The brain of PyFlowML.
    
    Pass in raw data + target column → get back a trained, evaluated model.
    Handles profiling, cleaning, preprocessing, model selection, and evaluation
    automatically.

    Example
    -------
    >>> engine = PyFlowEngine(df, target="Survived", time_limit=120)
    >>> engine.run()
    >>> engine.summary()
    """

    def __init__(self, df, target: str, time_limit: int = 120, metric: str = "auto"):
        self.df = df
        self.target = target
        self.time_limit = time_limit
        self.metric = metric
        self.profile_ = None
        self.best_model_ = None
        self.pipeline_ = None
        self.results_ = {}
        self._start_time = None

    def run(self):
        """Run the complete ML pipeline end-to-end."""
        self._start_time = time.time()
        logger.info("=" * 60)
        logger.info("  PyFlowML Engine — Starting Pipeline")
        logger.info("=" * 60)

        # Step 1: Profile
        with StepTracker("Data Profiling") as t:
            self.profile_ = DataProfiler(self.df, target=self.target).run()
            self.results_["profile"] = self.profile_
        logger.info(f"  Problem Type  : {self.profile_['problem_type']}")
        logger.info(f"  Dataset Shape : {self.df.shape}")

        # Step 2: Optimize memory
        with StepTracker("Memory Optimization") as t:
            self.df = MemoryOptimizer.reduce(self.df)
        logger.info(f"  Memory reduced by ~{t.memory_delta_mb:.1f} MB")

        # Step 3: Clean data
        with StepTracker("Data Cleaning") as t:
            recommendations = self.profile_.get("recommended_actions", {})
            cleaner = DataCleaner(self.df)
            cleaner.apply_recommendations(recommendations)
            self.df = cleaner.result()

        # Step 4: Split
        with StepTracker("Data Splitting") as t:
            splitter = DataSplitter(self.df, target=self.target)
            X_train, X_test, y_train, y_test = splitter.split()

        # Step 5: Preprocess
        with StepTracker("Preprocessing Pipeline") as t:
            self.pipeline_ = SmartPipeline()
            X_train = self.pipeline_.fit_transform(X_train, y_train)
            X_test = self.pipeline_.transform(X_test)

        # Step 6: Determine metric
        problem = self.profile_["problem_type"]
        if self.metric == "auto":
            self.metric = "f1" if problem == "classification" else "rmse"

        # Step 7: Train best model
        remaining_time = self.time_limit - (time.time() - self._start_time)
        with StepTracker("AutoML Training") as t:
            if problem == "classification":
                model = AutoClassifier(
                    metric=self.metric,
                    time_limit=max(10, int(remaining_time)),
                )
            else:
                model = AutoRegressor(
                    metric=self.metric,
                    time_limit=max(10, int(remaining_time)),
                )
            model.fit(X_train, y_train)
            self.best_model_ = model

        # Step 8: Evaluate
        with StepTracker("Evaluation") as t:
            if problem == "classification":
                report = Reporter.classification(model, X_test, y_test, print_report=False)
            else:
                report = Reporter.regression(model, X_test, y_test, print_report=False)
            self.results_["evaluation"] = report

        total_time = time.time() - self._start_time
        self.results_["total_time"] = total_time
        logger.info(f"\n  ✅ Pipeline completed in {total_time:.1f}s")
        self.summary()
        return self

    def summary(self):
        """Print a formatted summary of the run."""
        ev = self.results_.get("evaluation", {})
        prob = self.profile_["problem_type"] if self.profile_ else "unknown"
        best = self.best_model_.best_model_name_ if self.best_model_ else "N/A"

        print("\n" + "╔" + "═" * 50 + "╗")
        print("║{:^50}║".format("  PyFlowML — Results  "))
        print("╠" + "═" * 50 + "╣")
        print(f"║  Problem Type  : {prob:<32}║")
        print(f"║  Best Model    : {best:<32}║")
        for k, v in ev.items():
            if isinstance(v, float):
                print(f"║  {k:<16}: {v:<32.4f}║")
        print(f"║  Total Time    : {self.results_.get('total_time', 0):<30.1f}s║")
        print("╚" + "═" * 50 + "╝")

    def predict(self, X):
        """Predict using the best trained model."""
        if self.best_model_ is None:
            raise RuntimeError("Call .run() first.")
        X_transformed = self.pipeline_.transform(X)
        return self.best_model_.predict(X_transformed)
