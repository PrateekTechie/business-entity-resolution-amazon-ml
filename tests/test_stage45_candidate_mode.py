import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pipeline.disk_runner import DiskPipeline


class CandidateOnlyPipelineTests(unittest.TestCase):
    def test_empty_candidate_cache_fails_before_feature_or_training_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache = root / "upstream.sqlite"
            db = sqlite3.connect(cache)
            try:
                db.executescript("""
                    CREATE TABLE records(entity_id TEXT, source_id INTEGER);
                    CREATE TABLE candidates(source1_entity_id TEXT, target_entity_id TEXT, block_flags INTEGER);
                    CREATE TABLE features(source1_entity_id TEXT, target_entity_id TEXT);
                    CREATE TABLE truth(source1_entity_id TEXT, target_entity_id TEXT);
                    CREATE TABLE stage_state(stage TEXT, signature TEXT);
                    INSERT INTO records VALUES('S1-1',1),('S2-1',2);
                    INSERT INTO truth VALUES('S1-1','S2-1');
                """)
            finally:
                db.close()

            runner = DiskPipeline.__new__(DiskPipeline)
            runner.root = root
            runner.db_paths = {"train": root / "unused.sqlite"}
            with patch("src.pipeline.disk_runner.ResourceMonitor") as monitor:
                with patch.object(runner, "build_features") as build_features:
                    with patch.object(runner, "train_model") as train_model:
                        with self.assertRaisesRegex(ValueError, "No usable candidate cache"):
                            runner.run_from_candidates("train", cache)
            build_features.assert_not_called()
            train_model.assert_not_called()
            monitor.return_value.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
