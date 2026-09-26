import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.blocking.person3_retrieval import Person3CandidateBuilder, generate_candidate_tsv
from src.features.string_metrics import FeatureExtractor
from src.pipeline.disk_runner import DiskPipeline


class Person3RetrievalTests(unittest.TestCase):
    def _write_tsv(self, path: Path, headers, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(headers)
            writer.writerows(rows)
        return str(path)

    def _sample(self, root: Path):
        data = root / "data"
        data.mkdir(parents=True)
        source1 = self._write_tsv(data / "source1.tsv",
            ["entity_id", "business_name", "business_address", "country"], [
                ("S1-1", "Café & Sons Private Limited", "12 Rue de Paris", "France"),
                ("S1-2", "Mega Alpha Trading Limited", "", "US"),
                ("S1-3", "Odd DBA Brand Cömpany", "88 Old Road", "India"),
                ("S1-4", "Number Name Alpha Company", "77 North Road", "Canada"),
                ("S1-5", "Empty Address Brand LLC", "", "Atlantis"),
                ("S1-6", "One Off Lonely Entity", "", "France"),
            ])
        source2 = self._write_tsv(data / "source2.tsv",
            ["entity_id", "business_name", "business_address", "country"], [
                ("S2-1", "Cafe and Sons Pvt Ltd", "12 Rue de Paris", "France"),
                ("S2-2", "Trading Mega Alpha Ltd", "", "Germany"),
                ("S2-4", "Name Alpha Number Co", "77 South Road", "Japan"),
                ("S2-5", "Empty Address Brand Limited", "", "Atlantis"),
            ])
        source3 = self._write_tsv(data / "source3.tsv",
            ["entity_id", "business_name", "business_address", "country"], [
                ("S3-3", "Odd DBA Brand Company", "88 Old Road", "IN"),
            ])
        truth = self._write_tsv(data / "truth.tsv",
            ["source1_entity_id", "matched_entity_ids"], [
                ("S1-1", "S2-1"), ("S1-2", "S2-2"), ("S1-3", "S3-3"),
                ("S1-4", "S2-4"), ("S1-5", "S2-5"), ("S1-6", ""),
            ])
        return {"source1": source1, "source2": source2, "source3": source3}, truth

    def _builder(self, root: Path, **kwargs):
        options = {"chunk_rows": 2, "max_block_pairs": 10_000}
        options.update(kwargs)
        return Person3CandidateBuilder(root=root / "artifacts",
            mappings=Path(__file__).resolve().parents[1] / "config/text_mappings.json",
            **options)

    def test_normalized_key_families_cover_unicode_compact_order_and_suffix(self):
        builder = self._builder(Path(tempfile.gettempdir()))
        keys, _ = builder._keys_for_record("Café & Sons Private Limited", "12 Rue", "France")
        types = {kind for kind, _ in keys}
        values = {value for _, value in keys}
        self.assertTrue({1, 2, 3, 4, 5, 6, 10}.issubset(types))
        self.assertIn("cafe & sons private limited", {v for k, v in keys if k == 4})
        self.assertIn("cafésonsprivatelimited", {v for k, v in keys if k == 3})
        self.assertIn("café\x1fsons", {v for k, v in keys if k == 5})
        # Country is a key variant, never a filter for the shared base keys.
        other_country_keys, _ = builder._keys_for_record("Café & Sons Private Limited", "12 Rue", "Japan")
        self.assertEqual({v for k, v in keys if k not in {2, 12}},
                         {v for k, v in other_country_keys if k not in {2, 12}})

    def test_full_tiny_build_audits_recall_provenance_and_stage3_schema(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source_paths, truth = self._sample(root)
            builder = self._builder(root)
            db_path = root / "artifacts/train/candidate.sqlite"
            report = builder.build("train", source_paths, output_db=db_path, truth_path=truth)
            self.assertEqual(report["input_records"], {"source1_records": 6, "source2_records": 4, "source3_records": 1})
            self.assertEqual(report["recall_audit"]["pair_candidate_recall"], 1.0)
            self.assertEqual(report["recall_audit"]["missed_true_match_pairs"], 0)
            self.assertGreater(report["recall_audit"]["no_match_zero_candidate_entities"], 0)
            self.assertEqual(report["recall_audit"]["country_recall"]["india"]["recall"], 1.0)

            db = sqlite3.connect(db_path)
            try:
                rows = db.execute("SELECT source1_entity_id,target_entity_id,block_flags,retrieval_flags FROM candidates").fetchall()
                self.assertEqual(len(rows), len({(r[0], r[1]) for r in rows}))
                self.assertTrue(any(r[1].startswith("S2-") for r in rows))
                self.assertTrue(any(r[1].startswith("S3-") for r in rows))
                self.assertTrue(all(r[2] and r[3] for r in rows))
                feature_columns = [r[1] for r in db.execute("PRAGMA table_info(features)")]
                self.assertEqual(feature_columns[3:], FeatureExtractor.feature_columns())
                retrieval_kinds = {r[0] for r in db.execute(
                    "SELECT DISTINCT kind FROM retrieval_keys WHERE kind IN (16,17,18)")}
                self.assertEqual(retrieval_kinds, {16, 17, 18})

                # Exercise the actual Stage 3 reader against the Person 3 schema.
                runner = DiskPipeline.__new__(DiskPipeline)
                runner.feature_cols = FeatureExtractor.feature_columns()
                runner.feature_chunk_size = 3
                runner.extractor = FeatureExtractor(chunk_size=3)
                runner.db_paths = {"train": db_path}
                runner.force_rebuild = False
                runner.root = root
                self.assertEqual(runner.build_features(db, "train"), len(rows))
            finally:
                db.close()

            tsv = root / "candidate_pairs.tsv"
            output_stats = generate_candidate_tsv(db_path, tsv)
            self.assertEqual(output_stats["source1_rows"], 6)
            self.assertEqual(output_stats["invalid_target_ids"], 0)
            with tsv.open(encoding="utf-8", newline="") as stream:
                output_rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(output_rows), 6)
            self.assertEqual(next(r["candidate_entity_ids"] for r in output_rows
                                  if r["source1_entity_id"] == "S1-6"), "")

            # A second run must resume/reuse the completed cache deterministically.
            before_db = sqlite3.connect(db_path)
            pairs_before = before_db.execute(
                "SELECT source1_entity_id,target_entity_id,retrieval_flags FROM candidates ORDER BY 1,2").fetchall()
            before_db.close()
            builder.build("train", source_paths, output_db=db_path, truth_path=truth)
            check = sqlite3.connect(db_path)
            pairs_after = check.execute(
                "SELECT source1_entity_id,target_entity_id,retrieval_flags FROM candidates ORDER BY 1,2").fetchall()
            check.close()
            self.assertEqual(pairs_before, pairs_after)

    def test_chunk_checkpoint_resumes_after_interruption(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths, truth = self._sample(root)
            builder = self._builder(root)
            original_read_csv = __import__("src.blocking.person3_retrieval", fromlist=["pd"]).pd.read_csv

            def stop_after_first_chunk(*args, **kwargs):
                reader = original_read_csv(*args, **kwargs)
                try:
                    iterator = iter(reader)
                    yield next(iterator)
                    raise RuntimeError("simulated interruption after committed chunk")
                finally:
                    reader.close()

            db_path = root / "artifacts/train/resume.sqlite"
            with patch("src.blocking.person3_retrieval.pd.read_csv", side_effect=stop_after_first_chunk):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    builder.build("train", paths, output_db=db_path, truth_path=truth)
            resumed = builder.build("train", paths, output_db=db_path, truth_path=truth)
            self.assertEqual(resumed["input_records"]["source1_records"], 6)
            self.assertEqual(resumed["recall_audit"]["pair_candidate_recall"], 1.0)

    def test_oversized_duplicate_blocks_are_reported_and_misses_are_explained(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"
            data.mkdir()
            headers = ["entity_id", "business_name", "business_address", "country"]
            paths = {
                "source1": self._write_tsv(data / "s1.tsv", headers, [
                    ("S1-1", "Common Shop LLC", "10 Main Road", "US"),
                    ("S1-2", "Common Shop LLC", "10 Main Road", "US")]),
                "source2": self._write_tsv(data / "s2.tsv", headers, [
                    ("S2-1", "Common Shop LLC", "10 Main Road", "US"),
                    ("S2-2", "Common Shop LLC", "10 Main Road", "US")]),
                "source3": self._write_tsv(data / "s3.tsv", headers, []),
            }
            truth = self._write_tsv(data / "truth.tsv",
                ["source1_entity_id", "matched_entity_ids"], [("S1-1", "S2-1"), ("S1-2", "S2-2")])
            builder = self._builder(root, max_block_pairs=1)
            result = builder.build("train", paths, output_db=root / "artifacts/train/cap.sqlite", truth_path=truth)
            self.assertGreater(result["block_quality"]["blocks_over_pair_budget"], 0)
            self.assertGreater(result["block_quality"]["theoretical_pairs_omitted_by_budget"], 0)
            self.assertEqual(result["recall_audit"]["missed_true_match_pairs"], 2)
            misses = Path(result["recall_audit"]["misses_tsv"])
            with misses.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0]["raw_s1_name"])
            self.assertIn("configured pair budget", rows[0]["miss_reason"])

    def test_existing_unmarked_sqlite_database_is_never_modified(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "existing.sqlite"
            db = sqlite3.connect(path)
            db.execute("CREATE TABLE keep_me(value TEXT)")
            db.execute("INSERT INTO keep_me VALUES('unchanged')")
            db.commit()
            db.close()
            builder = self._builder(root)
            with self.assertRaisesRegex(ValueError, "Refusing to reuse"):
                builder._connect(path)
            db = sqlite3.connect(path)
            self.assertEqual(db.execute("SELECT value FROM keep_me").fetchone()[0], "unchanged")
            db.close()

    def test_secondary_country_keys_do_not_remove_open_set_base_keys(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths, truth = self._sample(root)
            builder = self._builder(root)
            db_path = root / "artifacts/train/secondary.sqlite"
            builder.build("train", paths, output_db=db_path, truth_path=truth)
            db = sqlite3.connect(db_path)
            try:
                for kind in (16, 17, 18):
                    self.assertGreater(db.execute(
                        "SELECT COUNT(*) FROM retrieval_keys WHERE kind=?", (kind,)).fetchone()[0], 0)
            finally:
                db.close()
            france, _ = builder._keys_for_record("A Unique Open Set Name", "19 Pine Road", "France")
            japan, _ = builder._keys_for_record("A Unique Open Set Name", "19 Pine Road", "Japan")
            self.assertIn((1, "a unique open set name"), france)
            self.assertIn((1, "a unique open set name"), japan)


if __name__ == "__main__":
    unittest.main()
