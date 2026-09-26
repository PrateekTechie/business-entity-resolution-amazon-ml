"""Training, entity-held-out validation and precision-weighted threshold tuning."""
from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

from src.features.string_metrics import FeatureExtractor


def prepare_training_labels(features_df: pd.DataFrame, ground_truth_tsv: str) -> pd.DataFrame:
    gt = pd.read_csv(ground_truth_tsv, sep="\t", dtype=str, keep_default_na=False,
                     usecols=["source1_entity_id", "matched_entity_ids"])
    pairs: set[tuple[str, str]] = set()
    match_counts: dict[str, int] = {}
    for s1, targets in zip(gt.source1_entity_id, gt.matched_entity_ids, strict=True):
        values = {target.strip() for target in targets.split(",") if target.strip()}
        pairs.update((s1, target) for target in values)
        match_counts[s1] = len(values)
    out = features_df.copy()
    out["label"] = np.fromiter(
        ((s1, target) in pairs for s1, target in zip(out.source1_entity_id, out.target_entity_id, strict=True)),
        dtype=np.int8, count=len(out)
    )
    out.attrs["ground_truth_match_counts"] = match_counts
    return out


def macro_f05_by_entity(frame: pd.DataFrame, probability: np.ndarray, threshold: float,
                        entity_ids: set[str], actual_counts: dict[str, int] | None = None) -> tuple[float, float, float]:
    work = pd.DataFrame({"entity": frame.source1_entity_id.to_numpy(),
                         "label": frame.label.to_numpy(), "pred": probability >= threshold})
    actual = work.groupby("entity").label.sum().reindex(entity_ids, fill_value=0).astype(float)
    if actual_counts is not None:
        actual = pd.Series({entity: float(actual_counts.get(entity, 0)) for entity in entity_ids})
    tp = work.loc[work.pred].groupby("entity").label.sum().reindex(entity_ids, fill_value=0).astype(float)
    predicted_count = work.groupby("entity").pred.sum().reindex(entity_ids, fill_value=0).astype(float)
    precision = tp.div(predicted_count.where(predicted_count > 0, 1))
    recall = tp.div(actual.where(actual > 0, 1))
    denominator = 0.25 * precision + recall
    f05 = 1.25 * precision * recall / denominator.replace(0, np.nan)
    empty_correct = actual.eq(0) & predicted_count.eq(0)
    precision = precision.mask(empty_correct, 1.0)
    recall = recall.mask(empty_correct, 1.0)
    f05 = f05.mask(empty_correct, 1.0).fillna(0.0)
    return float(precision.mean()), float(recall.mean()), float(f05.mean())


def train_xgboost(features_df: pd.DataFrame, save_model_path: str = "models_checkpoints/xgboost_matcher.pkl",
                  random_state: int = 42, all_entity_ids: set[str] | None = None) -> tuple[object, float, dict]:
    columns = FeatureExtractor.feature_columns()
    missing = set(columns + ["label", "source1_entity_id"]) - set(features_df.columns)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")
    all_ids = sorted(all_entity_ids) if all_entity_ids is not None else sorted(features_df.source1_entity_id.unique())
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_entity_idx, val_entity_idx = next(split.split(all_ids, groups=all_ids))
    train_ids = {all_ids[i] for i in train_entity_idx}
    val_ids = {all_ids[i] for i in val_entity_idx}
    train = features_df[features_df.source1_entity_id.isin(train_ids)]
    valid = features_df[features_df.source1_entity_id.isin(val_ids)]
    if train.label.nunique() != 2 or valid.label.nunique() != 2:
        raise ValueError("Training and validation must each contain positive and negative candidate pairs")
    weight = max(1.0, float((train.label == 0).sum()) / max(1, int((train.label == 1).sum())))
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.05, min_child_weight=2,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
        scale_pos_weight=weight, eval_metric="logloss", random_state=random_state,
        n_jobs=4, tree_method="hist"
    )
    model.fit(train[columns], train.label, eval_set=[(valid[columns], valid.label)], verbose=False)
    probabilities = model.predict_proba(valid[columns])[:, 1]
    ids = val_ids
    true_counts = features_df.attrs.get("ground_truth_match_counts", {})
    metrics = {}
    best_threshold, best_f05 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.99, 95):
        p, r, f05 = macro_f05_by_entity(valid, probabilities, float(threshold), ids, true_counts)
        metrics[float(threshold)] = {"precision": p, "recall": r, "macro_f0.5": f05}
        if f05 > best_f05:
            best_threshold, best_f05 = float(threshold), f05
    chosen = metrics[best_threshold]
    selected_mask = probabilities >= best_threshold
    chosen = {**chosen,
              "threshold": best_threshold,
              "predicted_matches": int(selected_mask.sum()),
              "entities_with_prediction": int(valid.loc[selected_mask, "source1_entity_id"].nunique()),
              "validation_entities": len(val_ids),
              "zero_match_predictions": int(len(val_ids) - valid.loc[selected_mask, "source1_entity_id"].nunique())}
    # Refit the deployment model on every labeled candidate after choosing the
    # decision threshold on the held-out entities.
    final_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=7, learning_rate=0.05, min_child_weight=2,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
        scale_pos_weight=max(1.0, float((features_df.label == 0).sum()) / max(1, int((features_df.label == 1).sum()))),
        eval_metric="logloss", random_state=random_state, n_jobs=4, tree_method="hist"
    )
    final_model.fit(features_df[columns], features_df.label, verbose=False)
    os.makedirs(os.path.dirname(save_model_path) or ".", exist_ok=True)
    joblib.dump({"model": final_model, "feature_columns": columns, "threshold": best_threshold}, save_model_path)
    print(f"Validation entity macro P/R/F0.5: {chosen['precision']:.4f}/{chosen['recall']:.4f}/{chosen['macro_f0.5']:.4f}; threshold={best_threshold:.2f}; predicted pairs={chosen['predicted_matches']:,}")
    return final_model, best_threshold, chosen
