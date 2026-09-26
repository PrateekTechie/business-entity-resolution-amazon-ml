import io
import json
import unittest
from unittest.mock import patch

import pandas as pd

from src.util.candidate_recall import evaluate_candidate_recall


class KeepOpenStringIO(io.StringIO):
    def close(self):
        self.flush()


class CandidateRecallTests(unittest.TestCase):
    def test_reports_pair_and_entity_recall_without_disk_artifacts(self):
        truth = pd.DataFrame({
            "source1_entity_id": ["S1-a", "S1-b"],
            "matched_entity_ids": ["S2-a,S3-a", ""],
        })
        candidates = pd.DataFrame({
            "source1_entity_id": ["S1-a", "S1-b"],
            "target_entity_id": ["S2-a", "S3-false"],
        })
        reference = pd.DataFrame({"entity_id": ["S1-a", "S1-b"],
                                  "clean_country": ["US", "France"]})
        saved = []

        def fake_open(*args, **kwargs):
            stream = KeepOpenStringIO()
            saved.append(stream)
            return stream

        with patch("src.util.candidate_recall.pd.read_csv", return_value=truth), \
                patch("src.util.candidate_recall.os.makedirs"), \
                patch("builtins.open", side_effect=fake_open):
            result = evaluate_candidate_recall(candidates, "unused.tsv", "reports/test.json", reference)
        self.assertEqual(result["candidate_recall"], 0.5)
        self.assertEqual(result["false_candidate_pairs"], 1)
        self.assertEqual(result["zero_retrieved_true_match_entities"], 0)
        self.assertEqual(result["country_recall"]["US"], 0.5)
        json.loads(saved[0].getvalue())
        self.assertIn("Candidate Recall Report", saved[1].getvalue())


if __name__ == "__main__":
    unittest.main()
