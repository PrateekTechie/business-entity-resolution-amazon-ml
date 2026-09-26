"""Prediction and official-format result aggregation."""
import joblib
import pandas as pd


def generate_matches(features_df: pd.DataFrame, model_path: str = "models_checkpoints/xgboost_matcher.pkl",
                     confidence_threshold: float | None = None,
                     all_source1_ids: list[str] | None = None) -> pd.DataFrame:
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        columns = artifact["feature_columns"]
        threshold = artifact["threshold"] if confidence_threshold is None else confidence_threshold
    else:  # Backward compatibility with the original bare-model checkpoint.
        model = artifact
        columns = [c for c in features_df.columns if c not in {"source1_entity_id", "target_entity_id", "label"}]
        threshold = 0.85 if confidence_threshold is None else confidence_threshold
    missing = set(columns) - set(features_df.columns)
    if missing:
        raise ValueError(f"Prediction features are missing: {sorted(missing)}")
    if len(features_df):
        probabilities = model.predict_proba(features_df[columns])[:, 1]
        selected = features_df.loc[probabilities >= threshold, ["source1_entity_id", "target_entity_id"]]
        mapping = selected.groupby("source1_entity_id").target_entity_id.apply(
            lambda ids: ",".join(sorted(set(map(str, ids))))
        ).to_dict()
    else:
        mapping = {}
    ids = all_source1_ids if all_source1_ids is not None else list(features_df.source1_entity_id.unique())
    return pd.DataFrame({"source1_entity_id": ids,
                         "matched_entity_ids": [mapping.get(s1, "") for s1 in ids]})
