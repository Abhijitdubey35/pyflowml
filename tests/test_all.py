"""
PyFlowML Test Suite — Core modules
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def clf_df():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [*iris.feature_names, "target"]
    return df


@pytest.fixture
def reg_df():
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame
    return df


@pytest.fixture
def dirty_df():
    df = pd.DataFrame({
        "age":    [25, None, 45, 25, None, 60],
        "income": [50000, 80000, 30000, 50000, 70000, 90000],
        "city":   ["NY", "LA", "NY", "NY", "LA", "Chicago"],
        "target": [1, 0, 1, 1, 0, 1],
    })
    return df


# ─── Data Module Tests ─────────────────────────────────────────────────────────

class TestDataLoader:
    def test_from_dataframe(self, clf_df):
        from pyflowml.data.loader import DataLoader
        result = DataLoader.from_dataframe(clf_df)
        assert result.shape == clf_df.shape

    def test_from_dict(self):
        from pyflowml.data.loader import DataLoader
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        df = DataLoader.from_dict(data)
        assert df.shape == (3, 2)
        assert list(df.columns) == ["a", "b"]


class TestDataCleaner:
    def test_handle_nulls_median(self, dirty_df):
        from pyflowml.data.cleaner import DataCleaner
        cleaner = DataCleaner(dirty_df)
        result = cleaner.handle_nulls(strategy="median").result()
        assert result["age"].isna().sum() == 0

    def test_remove_duplicates(self, dirty_df):
        from pyflowml.data.cleaner import DataCleaner
        result = DataCleaner(dirty_df).remove_duplicates().result()
        assert len(result) <= len(dirty_df)

    def test_encode_categoricals(self, dirty_df):
        from pyflowml.data.cleaner import DataCleaner
        result = DataCleaner(dirty_df).encode_categoricals(method="label").result()
        assert result["city"].dtype in [np.int32, np.int64, int]

    def test_chain(self, dirty_df):
        from pyflowml.data.cleaner import DataCleaner
        result = (DataCleaner(dirty_df)
                  .handle_nulls()
                  .remove_duplicates()
                  .result())
        assert result.isna().sum().sum() == 0


class TestMemoryOptimizer:
    def test_reduce(self, clf_df):
        from pyflowml.data.memory import MemoryOptimizer
        reduced = MemoryOptimizer.reduce(clf_df)
        before = clf_df.memory_usage(deep=True).sum()
        after = reduced.memory_usage(deep=True).sum()
        assert after <= before

    def test_audit_returns_df(self, clf_df):
        from pyflowml.data.memory import MemoryOptimizer
        audit = MemoryOptimizer.audit(clf_df)
        assert "column" in audit.columns
        assert "memory_mb" in audit.columns


class TestDataSplitter:
    def test_split_shapes(self, clf_df):
        from pyflowml.data.splitter import DataSplitter
        X_train, X_test, y_train, y_test = DataSplitter(clf_df, target="target").split()
        assert len(X_train) + len(X_test) == len(clf_df)

    def test_three_way_split(self, clf_df):
        from pyflowml.data.splitter import DataSplitter
        result = DataSplitter(clf_df, target="target", val_size=0.1).split_three_way()
        assert len(result) == 6
        total = sum(len(r) for r in result[:3])
        assert total == len(clf_df)


# ─── Preprocessing Tests ───────────────────────────────────────────────────────

class TestScaler:
    def test_standard_scaler(self, clf_df):
        from pyflowml.preprocessing.scaler import Scaler
        X = clf_df.drop(columns=["target"])
        scaler = Scaler("standard")
        X_scaled = scaler.fit_transform(X)
        assert abs(X_scaled.mean().mean()) < 0.1

    def test_preserves_columns(self, clf_df):
        from pyflowml.preprocessing.scaler import Scaler
        X = clf_df.drop(columns=["target"])
        scaler = Scaler("minmax")
        X_scaled = scaler.fit_transform(X)
        assert list(X_scaled.columns) == list(X.columns)


class TestFeatureSelector:
    def test_drops_constant(self):
        from pyflowml.preprocessing.feature_selector import FeatureSelector
        df = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5], "c": [7, 8, 9]})
        selector = FeatureSelector(method="constant")
        result = selector.fit_transform(df)
        assert "b" not in result.columns

    def test_drops_correlated(self):
        from pyflowml.preprocessing.feature_selector import FeatureSelector
        np.random.seed(0)
        x = np.random.randn(100)
        df = pd.DataFrame({"a": x, "b": x * 1.001, "c": np.random.randn(100)})
        selector = FeatureSelector(method="correlation", threshold=0.9)
        result = selector.fit_transform(df)
        assert result.shape[1] < df.shape[1]


class TestSmartPipeline:
    def test_consistent_categorical_encoding(self):
        """Train and test must share the SAME category→code mapping."""
        from pyflowml.preprocessing.pipeline import SmartPipeline
        train = pd.DataFrame({
            "city": ["NY", "LA", "SF", "NY", "LA", "SF"],
            "val":  [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        })
        test = pd.DataFrame({
            "city": ["LA", "NY", "SF"],
            "val":  [2.0, 1.0, 3.0],
        })
        pipe = SmartPipeline(scaler_method="standard", selector_method="all")
        Xtr = pipe.fit_transform(train)
        Xte = pipe.transform(test)
        train_map = dict(zip(train["city"], Xtr["city"]))
        for city, code in zip(test["city"], Xte["city"]):
            assert train_map[city] == code

    def test_unseen_category_is_safe(self):
        """Categories unseen during fit encode to -1 instead of crashing."""
        from pyflowml.preprocessing.pipeline import SmartPipeline
        train = pd.DataFrame({"c": ["a", "b", "a", "b"], "x": [1.0, 2.0, 3.0, 4.0]})
        test = pd.DataFrame({"c": ["a", "zzz"], "x": [1.0, 2.0]})
        pipe = SmartPipeline(selector_method="all")
        pipe.fit_transform(train)
        out = pipe.transform(test)
        assert out["c"].iloc[1] == -1


# ─── Model Tests ───────────────────────────────────────────────────────────────

class TestAutoClassifier:
    def test_fit_predict(self, clf_df):
        from pyflowml.data.splitter import DataSplitter
        from pyflowml.models.auto import AutoClassifier
        X_train, X_test, y_train, y_test = DataSplitter(clf_df, target="target").split()
        clf = AutoClassifier(metric="f1", time_limit=30)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert len(preds) == len(y_test)

    def test_leaderboard(self, clf_df):
        from pyflowml.data.splitter import DataSplitter
        from pyflowml.models.auto import AutoClassifier
        X_train, X_test, y_train, y_test = DataSplitter(clf_df, target="target").split()
        clf = AutoClassifier(metric="accuracy", time_limit=20)
        clf.fit(X_train, y_train)
        lb = clf.leaderboard()
        assert len(lb) > 0
        assert "model" in lb.columns


class TestAutoRegressor:
    def test_fit_predict(self, reg_df):
        from pyflowml.data.splitter import DataSplitter
        from pyflowml.models.auto import AutoRegressor
        X_train, X_test, y_train, y_test = DataSplitter(reg_df, target="target").split()
        reg = AutoRegressor(metric="r2", time_limit=20)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        assert len(preds) == len(y_test)


class TestBudget:
    def test_tiny_budget_still_returns_model(self, clf_df):
        """Admission control must never leave .fit() without a usable model."""
        from pyflowml.data.splitter import DataSplitter
        from pyflowml.models.auto import AutoClassifier
        X_train, X_test, y_train, y_test = DataSplitter(clf_df, target="target").split()
        clf = AutoClassifier(metric="accuracy", time_limit=1, random_state=0)
        clf.fit(X_train, y_train)
        assert clf.best_model_ is not None
        assert len(clf._leaderboard) >= 1
        assert len(clf.predict(X_test)) == len(y_test)


class TestConsole:
    def test_ensure_utf8_console_is_idempotent_and_safe(self):
        from pyflowml.utils.console import ensure_utf8_console
        ensure_utf8_console()
        ensure_utf8_console()  # second call must be a no-op and never raise


class TestCLISubcommands:
    def test_clean_writes_csv(self, tmp_path, dirty_df):
        import pyflowml.cli as cli
        src = tmp_path / "dirty.csv"
        dirty_df.to_csv(src, index=False)
        out = tmp_path / "clean.csv"
        cli.main(["clean", str(src), "-o", str(out)])
        assert out.exists()
        cleaned = pd.read_csv(out)
        assert cleaned.isna().sum().sum() == 0   # nulls handled on export

    def test_predict_writes_predictions(self, tmp_path, clf_df):
        import pyflowml.cli as cli
        from pyflowml.preprocessing.pipeline import SmartPipeline
        from pyflowml.models.auto import AutoClassifier
        from pyflowml.utils.saver import ModelSaver
        X = clf_df.drop(columns=["target"]); y = clf_df["target"]
        pipe = SmartPipeline(selector_method="all")
        Xt = pipe.fit_transform(X, y)
        clf = AutoClassifier(metric="accuracy", time_limit=1, random_state=0)
        clf.fit(Xt, y)
        model_path = ModelSaver.save(clf, "m", directory=str(tmp_path), pipeline=pipe)

        newcsv = tmp_path / "new.csv"; X.head(5).to_csv(newcsv, index=False)
        out = tmp_path / "preds.csv"
        cli.main(["predict", model_path, str(newcsv), "-o", str(out)])
        assert out.exists()
        preds = pd.read_csv(out)
        assert "prediction" in preds.columns and len(preds) == 5

    def test_bare_invocation_routes_to_full_pipeline(self, monkeypatch):
        """`pyflowml` with no sub-command must hit the interactive pipeline path."""
        import pyflowml.cli as cli
        called = {}
        monkeypatch.setattr(cli, "_interactive_pipeline", lambda argv: called.setdefault("argv", argv))
        cli.main([])
        assert called.get("argv") == []


# ─── Evaluation Tests ──────────────────────────────────────────────────────────

class TestReporter:
    def test_classification_report(self, clf_df):
        from pyflowml.data.splitter import DataSplitter
        from pyflowml.models.auto import AutoClassifier
        from pyflowml.evaluation.reporter import Reporter
        X_train, X_test, y_train, y_test = DataSplitter(clf_df, target="target").split()
        clf = AutoClassifier(metric="f1", time_limit=20)
        clf.fit(X_train, y_train)
        metrics = Reporter.classification(clf, X_test, y_test, print_report=True)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


# ─── NLP Tests ────────────────────────────────────────────────────────────────

class TestTextCleaner:
    def test_basic_cleaning(self):
        from pyflowml.nlp.text_cleaner import TextCleaner
        cleaner = TextCleaner(remove_stopwords=False, lemmatize=False)
        result = cleaner.fit_transform(["Hello World!", "  Test 123  "])
        assert result[0] == "hello world"

    def test_removes_html(self):
        from pyflowml.nlp.text_cleaner import TextCleaner
        cleaner = TextCleaner(remove_stopwords=False, lemmatize=False)
        result = cleaner.fit_transform(["<b>Hello</b> world"])
        assert "<b>" not in result[0]


class TestVectorizer:
    def test_tfidf(self):
        from pyflowml.nlp.vectorizer import Vectorizer
        texts = ["hello world", "machine learning", "deep learning rocks"]
        vec = Vectorizer(method="tfidf", max_features=100)
        X = vec.fit_transform(texts)
        assert X.shape[0] == 3

    def test_bow(self):
        from pyflowml.nlp.vectorizer import Vectorizer
        texts = ["hello world", "machine learning", "deep learning rocks",
                 "natural language processing", "neural networks are great"]
        vec = Vectorizer(method="bow", max_features=50, min_df=1)
        X = vec.fit_transform(texts)
        assert X.shape[0] == 5
