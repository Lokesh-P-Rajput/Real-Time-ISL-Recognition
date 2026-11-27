"""STEP 2 - Train XGBoost, Random Forest, SVM and KNN, compare them and save one model.

    python train_model.py                      saves XGBoost (default, see DEFAULT_MODEL in isl/config.py)
    python train_model.py --model auto         saves the most accurate model (the fastest if several are within 1 %)
    python train_model.py --model rf           xgboost | rf | svm | knn | auto
    python train_model.py --split-by-session   stricter TEST only: whole recording sessions are held out.
                                               It does not replace the saved model; reports go to reports/session_split/
    python train_model.py --cv 0               skip cross-validation (faster)

Outputs
    models/isl_model.joblib            the saved model (loaded by the app)
    reports/model_comparison.csv       accuracy / precision / recall / F1 / prediction time of all models
    reports/classification_report.txt  per-gesture precision, recall, F1 of the saved model
    reports/confusion_matrix.png       confusion matrix of the saved model
    reports/metrics.json               summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,  # noqa: E402
                             classification_report, precision_recall_fscore_support)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedKFold,  # noqa: E402
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from isl import config  # noqa: E402
from isl.features import FEATURE_NAMES  # noqa: E402

MODEL_ALIASES = {"xgboost": "XGBoost", "xgb": "XGBoost", "rf": "Random Forest",
                 "randomforest": "Random Forest", "svm": "SVM", "knn": "KNN", "auto": "auto"}


def resolve_model_choice(choice: str | None) -> str:
    key = (choice or config.DEFAULT_MODEL).lower().replace(" ", "").replace("_", "").replace("-", "")
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unknown model '{choice}'. Use one of: xgboost, rf, svm, knn, auto.")
    return MODEL_ALIASES[key]


def make_xgboost():
    try:
        from xgboost import XGBClassifier
    except Exception as exc:       # not installed, or the Windows C++ runtime is missing
        raise RuntimeError(
            f"XGBoost could not be loaded ({type(exc).__name__}). Install it with  pip install xgboost  "
            "and, on Windows, also the Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe"
        ) from exc
    return XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.9,
                         colsample_bytree=0.9, tree_method="hist",
                         random_state=config.RANDOM_STATE, verbosity=0)


def build_candidates(strict_xgboost: bool = True) -> dict:
    candidates = {}
    try:
        candidates["XGBoost"] = make_xgboost()
    except RuntimeError as exc:
        if strict_xgboost:
            raise
        print(f"Warning: {exc}\n  XGBoost is skipped in this run.")
    candidates["Random Forest"] = RandomForestClassifier(n_estimators=150, random_state=config.RANDOM_STATE)
    candidates["SVM"] = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=config.RANDOM_STATE)
    candidates["KNN"] = KNeighborsClassifier(n_neighbors=5, weights="distance")
    return candidates


def load_dataset(csv_path) -> pd.DataFrame:
    """Read the CSV and remove incorrect / incomplete / duplicate samples."""
    df = pd.read_csv(csv_path)
    missing = [c for c in ["label"] + FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing[:5]} ...")
    if "session" not in df.columns:
        df["session"] = "unknown"
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["label"] + FEATURE_NAMES)
    df = df.drop_duplicates(subset=["label"] + FEATURE_NAMES).reset_index(drop=True)
    return df


def time_single_prediction(pipe, X_test, n: int = 100) -> float:
    """Average time (ms) to classify ONE sample - what the live app really does."""
    rows = X_test[: min(n, len(X_test))]
    start = perf_counter()
    for row in rows:
        pipe.predict_proba(row.reshape(1, -1))
    return (perf_counter() - start) / len(rows) * 1000.0


def train(dataset_csv=None, model_path=None, report_dir=None, test_size=None,
          split_by_session=False, cv=5, model_choice=None, save_model=True, verbose=True) -> dict:
    dataset_csv = dataset_csv or config.DATASET_CSV
    model_path = Path(model_path or config.MODEL_PATH)
    report_dir = Path(report_dir or config.REPORT_DIR)
    test_size = config.TEST_SIZE if test_size is None else test_size
    chosen = resolve_model_choice(model_choice)
    log = print if verbose else (lambda *a, **k: None)

    df = load_dataset(dataset_csv)
    allowed = set(config.GESTURES) | {config.NEUTRAL_LABEL}
    leftovers = df.loc[~df["label"].isin(allowed), "label"].value_counts()
    if not leftovers.empty and df["label"].isin(allowed).any():
        log("Ignoring recorded samples of signs that are not in assets/sign_guide.csv any more: "
            + ", ".join(f"{name} ({count})" for name, count in leftovers.items()))
        df = df[df["label"].isin(allowed)].reset_index(drop=True)
    counts = df["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("Need samples of at least 2 gestures. Run collect_data.py first.")
    if counts.min() < 10:
        raise ValueError(f"'{counts.idxmin()}' has only {counts.min()} samples - record more (aim for 300+).")

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    n_classes = len(encoder.classes_)

    if split_by_session and df["session"].nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=config.RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups=df["session"]))
        split_mode = "session-held-out"
    else:
        if split_by_session:
            log("Only one recording session found - using the normal random split instead.")
        train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=test_size,
                                               stratify=y, random_state=config.RANDOM_STATE)
        split_mode = "random-stratified"
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    absent = sorted(set(range(n_classes)) - set(np.unique(y_train)))
    if absent:
        names = ", ".join(str(encoder.classes_[i]) for i in absent)
        raise ValueError(f"No training samples of: {names} in this split. Record those gestures in more than "
                         "one session, or train without --split-by-session.")

    log(f"\nSamples: {len(df)} | gestures: {n_classes} | train: {len(X_train)} | test: {len(X_test)} | split: {split_mode}")

    folds = min(int(cv), int(np.bincount(y_train).min()))

    rows, fitted = [], {}
    for name, clf in build_candidates(strict_xgboost=(chosen == "XGBoost")).items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        cv_acc = np.nan
        if folds >= 2:
            cv_acc = cross_val_score(pipe, X_train, y_train, scoring="accuracy",
                                     cv=StratifiedKFold(folds, shuffle=True, random_state=config.RANDOM_STATE)).mean()
        start = perf_counter()
        pipe.fit(X_train, y_train)
        fit_s = perf_counter() - start
        y_pred = pipe.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "CV accuracy": cv_acc,
            "Training time (s)": fit_s,
            "Prediction time (ms)": time_single_prediction(pipe, X_test),
        })
        fitted[name] = (pipe, y_pred)
        log(f"  trained {name:<14} accuracy = {rows[-1]['Accuracy']:.4f}")

    results = pd.DataFrame(rows)
    top_accuracy = results["Accuracy"].max()
    most_accurate = results.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
    # "auto": best accuracy wins; among models within 1 % of it, the fastest one is taken.
    auto_name = results[results["Accuracy"] >= top_accuracy - 0.01].sort_values("Prediction time (ms)").iloc[0]["Model"]
    if chosen == "auto":
        best_name, reason = auto_name, "auto (most accurate; fastest among near-equal models)"
    else:
        best_name, reason = chosen, f"chosen with --model / DEFAULT_MODEL ({chosen})"
    best_pipe, best_pred = fitted[best_name]

    results.insert(1, "Selected", ["yes" if m == best_name else "" for m in results["Model"]])
    results = results.round(4)
    log("\n" + results.to_string(index=False))
    log(f"\nSelected model: {best_name}  [{reason}]")
    best_accuracy = float(results.loc[results["Model"] == best_name, "Accuracy"].iloc[0])
    if best_accuracy < top_accuracy - 0.01:
        log(f"Note: {most_accurate} scored higher on this data ({top_accuracy:.4f} vs {best_accuracy:.4f}). "
            "Run  python train_model.py --model auto  to let the program pick.")

    # ----------------------------------------------------------- save model + reports
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    meta = {"model_name": best_name, "classes": [str(c) for c in encoder.classes_],
            "test_accuracy": float(accuracy_score(y_test, best_pred)), "split": split_mode,
            "n_samples": int(len(df))}
    if save_model:
        joblib.dump({"model": best_pipe, "label_encoder": encoder, "meta": meta}, model_path)

    results.to_csv(report_dir / "model_comparison.csv", index=False)
    labels = np.arange(n_classes)
    (report_dir / "classification_report.txt").write_text(
        classification_report(y_test, best_pred, labels=labels, target_names=encoder.classes_, zero_division=0),
        encoding="utf-8")

    fig, ax = plt.subplots(figsize=(max(6, n_classes * 0.65), max(5, n_classes * 0.6)))
    ConfusionMatrixDisplay.from_predictions(y_test, best_pred, labels=labels, display_labels=encoder.classes_,
                                            xticks_rotation=45, cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"Confusion matrix - {best_name} (test set)")
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    summary = {**meta, "best_model": best_name, "selection": reason, "most_accurate_model": most_accurate,
               "models": results.to_dict(orient="records")}
    (report_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if save_model:
        log(f"\nSaved model   -> {model_path}")
    else:
        log("\nEvaluation only: the saved model was NOT changed.")
    log(f"Saved reports -> {report_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ISL gesture classifier")
    parser.add_argument("--model", default=config.DEFAULT_MODEL,
                        help="xgboost | rf | svm | knn | auto  (default: %(default)s)")
    parser.add_argument("--split-by-session", action="store_true",
                        help="hold out whole recording sessions for testing (stricter)")
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    parser.add_argument("--cv", type=int, default=5, help="cross-validation folds (0 = skip)")
    args = parser.parse_args()
    if not config.DATASET_CSV.exists():
        sys.exit("data/dataset.csv not found. Run  python collect_data.py  first.")
    try:
        strict = args.split_by_session           # a stricter test never replaces the saved model
        train(test_size=args.test_size, split_by_session=strict, cv=args.cv, model_choice=args.model,
              save_model=not strict,
              report_dir=config.REPORT_DIR / "session_split" if strict else None)
    except (ValueError, RuntimeError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
