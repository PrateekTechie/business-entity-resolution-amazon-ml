import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

from src.features.string_metrics import FeatureExtractor
from src.models.predict import generate_matches
from src.models.train import macro_f05_by_entity, prepare_training_labels, train_xgboost


class DummyModel:
    def predict_proba(self, matrix):
        return np.asarray([[0.01, 0.99] if value > 0.5 else [0.99, 0.01]
                           for value in matrix["name_ratio"]])


class FeatureAndModelTests(unittest.TestCase):
    def test_candidate_features_are_numeric_and_complete(self):
        ref = pd.DataFrame({"entity_id": ["S1-a"], "clean_name": ["acme shop"],
                            "clean_address": ["1 main road"], "clean_country": ["france"]})
        target = pd.DataFrame({"entity_id": ["S2-a"], "clean_name": ["acme shop"],
                               "clean_address": ["1 main road"], "clean_country": ["france"]})
        pairs = pd.DataFrame({"source1_entity_id": ["S1-a"], "target_entity_id": ["S2-a"]})
        result = FeatureExtractor(chunk_size=1).generate_candidate_features(pairs, ref, target)
        self.assertEqual(result.loc[0, "name_exact"], 1.0)
        self.assertEqual(result.loc[0, "addr_exact"], 1.0)
        self.assertEqual(result.loc[0, "same_country"], 1.0)
        self.assertTrue(np.isfinite(result[FeatureExtractor.feature_columns()].to_numpy()).all())

    def test_feature_streaming_and_edge_cases(self):
        ref = pd.DataFrame({
            "entity_id": ["S1-1", "S1-2", "S1-3", "S1-4"],
            "clean_name": ["Acme Bakery", "North Star Trading", "", "Acme Bakery"],
            "clean_address": ["12 Main Street 90210", "", "", "12 Main Street"],
            "clean_country": ["france"] * 4,
        })
        target = pd.DataFrame({
            "entity_id": ["S2-1", "S2-2", "S2-3", "S3-4"],
            "clean_name": ["Acme Bakery", "Trading North Star", "", "Acme Backery"],
            "clean_address": ["12 Main Street 90210", "", "", "12 Main Street"],
            "clean_country": ["france"] * 4,
        })
        pairs = pd.DataFrame({"source1_entity_id": ["S1-1", "S1-2", "S1-3", "S1-4"],
                              "target_entity_id": ["S2-1", "S2-2", "S2-3", "S3-4"]})
        extractor = FeatureExtractor(chunk_size=2)
        chunks = list(extractor.iter_candidate_features(pairs, ref, target))
        self.assertEqual([len(chunk) for chunk in chunks], [2, 2])
        result = pd.concat(chunks, ignore_index=True)
        self.assertEqual(result.loc[0, "name_exact"], 1.0)
        self.assertEqual(result.loc[0, "addr_exact"], 1.0)
        self.assertEqual(result.loc[1, "name_token_sort_ratio"], 1.0)
        self.assertGreater(result.loc[3, "name_ratio"], 0.8)
        self.assertEqual(result.loc[2, "missing_name"], 1.0)
        self.assertEqual(result.loc[2, "missing_address"], 1.0)
        self.assertEqual(result.loc[1, "addr_informative_shared_tokens"], 0.0)
        self.assertTrue(np.isfinite(result[FeatureExtractor.feature_columns()].to_numpy()).all())
        with self.assertRaisesRegex(ValueError, "iter_candidate_features"):
            extractor.generate_candidate_features(pairs, ref, target)

    def test_ground_truth_parsing_and_singleton_metric(self):
        features = pd.DataFrame({"source1_entity_id": ["S1-a", "S1-b"],
                                 "target_entity_id": ["S2-a", "S3-b"]})
        truth = pd.DataFrame({"source1_entity_id": ["S1-a", "S1-b"],
                              "matched_entity_ids": ["S2-a,S3-a", ""]})
        with patch("src.models.train.pd.read_csv", return_value=truth):
            labeled = prepare_training_labels(features, "unused.tsv")
        self.assertEqual(labeled.label.tolist(), [1, 0])
        metric_frame = pd.DataFrame({"source1_entity_id": ["S1-a", "S1-b"], "label": [1, 0]})
        p, r, f = macro_f05_by_entity(metric_frame, np.array([0.99, 0.01]), .5,
                                      {"S1-a", "S1-b"}, {"S1-a": 1, "S1-b": 0})
        self.assertEqual((p, r, f), (1.0, 1.0, 1.0))

    def test_prediction_preserves_zero_and_multiple_matches(self):
        frame = pd.DataFrame({"source1_entity_id": ["S1-a", "S1-a"],
                              "target_entity_id": ["S3-z", "S2-a"], "name_ratio": [.99, .99]})
        artifact = {"model": DummyModel(), "feature_columns": ["name_ratio"], "threshold": .5}
        with patch("src.models.predict.joblib.load", return_value=artifact):
            actual = generate_matches(frame, "unused.pkl", all_source1_ids=["S1-a", "S1-empty"])
        self.assertEqual(actual.matched_entity_ids.tolist(), ["S2-a,S3-z", ""])

    def test_entity_held_out_training_and_threshold_tuning(self):
        rows = []
        counts = {}
        for i in range(30):
            counts[f"S1-{i}"] = 1
            for j in range(3):
                row = {column: 0.0 for column in FeatureExtractor.feature_columns()}
                row.update(source1_entity_id=f"S1-{i}", target_entity_id=f"S2-{i}-{j}", label=int(j == 0))
                row["name_ratio"] = .98 if j == 0 else .05 + .02 * j
                rows.append(row)
        frame = pd.DataFrame(rows)
        frame.attrs["ground_truth_match_counts"] = counts
        with patch("src.models.train.joblib.dump"):
            model, threshold, metrics = train_xgboost(frame, "models_checkpoints/test-model.pkl", 13, set(counts))
        self.assertGreaterEqual(threshold, .05)
        self.assertLessEqual(threshold, .99)
        self.assertEqual(metrics["macro_f0.5"], 1.0)
        self.assertGreater(metrics["validation_entities"], 0)
        self.assertEqual(model.get_params()["scale_pos_weight"], 2.0)


if __name__ == "__main__":
    unittest.main()
