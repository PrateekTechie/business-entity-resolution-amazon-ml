"""Disk-backed, resumable entity-resolution runner for the full challenge data.

SQLite is used as an indexed local spool so neither all candidate pairs nor all
features need to live in Python memory. XGBoost reads numeric feature batches via
external-memory QuantileDMatrix iterators.
"""
from __future__ import annotations

import csv
import threading
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import xgboost as xgb

from src.features.string_metrics import FeatureExtractor
from src.preprocessing.text_cleaner import TextCleaner


WORD_RE = re.compile(r"[\w]+", re.UNICODE)
NUMBER_RE = re.compile(r"\b\d{2,8}\b")
STOP = {"THE", "INC", "LLC", "LTD", "LIMITED", "CORP", "CORPORATION", "GROUP", "COMPANY", "CO",
        "GLOBAL", "INTERNATIONAL", "SERVICES", "HOLDINGS", "ENTERPRISES", "ASSOCIATES", "SOLUTIONS",
        "AND", "PTE", "PVT", "PRIVATE", "GMBH", "SA", "BV", "SL", "AG", "TECHNOLOGIES",
        "STREET", "ROAD", "RD", "ST", "AVENUE", "AVE", "LANE", "DRIVE", "DR", "NEAR", "OPPOSITE"}
BLOCK_BITS = {"exact_name": 1, "name_token": 2, "name_prefix": 4, "address_number": 8,
              "exact_address": 16, "address_token": 32}
BLOCK_TYPES = list(BLOCK_BITS)


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


class ResourceMonitor:
    """Sample process RSS and D: free space while a pipeline stage is active."""
    def __init__(self, root: Path, stage: str):
        self.root, self.stage = root, stage
        self.started = time.perf_counter()
        self.start_rss = _rss_mb()
        self.peak_rss = self.start_rss
        self.start_free = shutil.disk_usage(root).free
        self.minimum_free = self.start_free
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.wait(1.0):
            self.peak_rss = max(self.peak_rss, _rss_mb())
            self.minimum_free = min(self.minimum_free, shutil.disk_usage(self.root).free)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.peak_rss = max(self.peak_rss, _rss_mb())
        self.minimum_free = min(self.minimum_free, shutil.disk_usage(self.root).free)
        result = {"stage": self.stage, "elapsed_seconds": time.perf_counter()-self.started,
                  "rss_start_mib": self.start_rss, "rss_observed_peak_mib": self.peak_rss,
                  "d_free_start_gib": self.start_free/1024**3,
                  "d_free_min_observed_gib": self.minimum_free/1024**3,
                  "d_free_end_gib": shutil.disk_usage(self.root).free/1024**3,
                  "peak_is_sampled_not_absolute": True}
        path = self.root / "artifacts/reports" / f"performance_{self.stage}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[{self.stage}] elapsed={result['elapsed_seconds']:.1f}s; sampled peak RSS={self.peak_rss:.0f} MiB; D free={result['d_free_end_gib']:.1f} GiB")
        return result


class DiskPipeline:
    def __init__(self, config: dict[str, Any], project_root: Path, *, force_rebuild: bool = False,
                 reuse_candidates: bool = False, reuse_features: bool = False):
        self.config = config
        self.root = Path(config["storage"]["root"]).resolve()
        self.project_root = project_root.resolve()
        self.force_rebuild = force_rebuild
        self.reuse_candidates = reuse_candidates
        self.reuse_features = reuse_features
        self.chunk_size = int(config["storage"].get("chunk_size", 50_000))
        self.feature_chunk_size = int(config["features"].get("chunk_size", 25_000))
        self.max_group_size = int(config["blocking"].get("max_group_size", 300))
        self.paths = config["data_paths"]
        for folder in ["data", "artifacts/candidates/train", "artifacts/candidates/test",
                       "artifacts/features/train", "artifacts/features/test", "artifacts/models",
                       "artifacts/reports", "artifacts/cache", "artifacts/temp", "submission/output",
                       "code"]:
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        self.temp = Path(config["storage"].get("temp_dir", self.root / "artifacts/temp"))
        self.temp.mkdir(parents=True, exist_ok=True)
        os.environ["TEMP"] = str(self.temp)
        os.environ["TMP"] = str(self.temp)
        import tempfile
        tempfile.tempdir = str(self.temp)
        usage = shutil.disk_usage(self.root)
        if usage.free < 10 * 1024**3:
            raise OSError(f"Only {usage.free / 1024**3:.2f} GiB free under {self.root}; need at least 10 GiB safety space")
        self.db_paths = {name: self.root / "artifacts/cache" / f"{name}.sqlite" for name in ("train", "test", "benchmark")}
        self.cleaner = TextCleaner(str((self.project_root / "config/text_mappings.json").resolve()))
        self.extractor = FeatureExtractor(self.feature_chunk_size)
        self.feature_cols = self.extractor.feature_columns()

    def _connect(self, split: str) -> sqlite3.Connection:
        path = self.db_paths[split]
        if self.force_rebuild and path.exists():
            path.unlink()
            for sidecar in (Path(str(path) + "-wal"), Path(str(path) + "-shm")):
                sidecar.unlink(missing_ok=True)
        conn = sqlite3.connect(path, timeout=120)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("PRAGMA cache_size=-262144")  # ~256 MiB page cache
        conn.execute("PRAGMA mmap_size=268435456")
        conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema(conn)
        return conn

    def _create_schema(self, db: sqlite3.Connection) -> None:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS records(
              entity_id TEXT PRIMARY KEY, source_id INTEGER NOT NULL,
              clean_name TEXT NOT NULL, clean_address TEXT NOT NULL, clean_country TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS records_source ON records(source_id,entity_id);
            CREATE TABLE IF NOT EXISTS block_keys(
              kind INTEGER NOT NULL, block_key TEXT NOT NULL, side INTEGER NOT NULL,
              entity_id TEXT NOT NULL, PRIMARY KEY(kind,block_key,entity_id),
              FOREIGN KEY(entity_id) REFERENCES records(entity_id) ON DELETE CASCADE
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS block_lookup ON block_keys(kind,block_key,side,entity_id);
            CREATE TABLE IF NOT EXISTS stage_state(stage TEXT PRIMARY KEY, signature TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS candidates(
              source1_entity_id TEXT NOT NULL, target_entity_id TEXT NOT NULL, block_flags INTEGER NOT NULL,
              PRIMARY KEY(source1_entity_id,target_entity_id)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS candidates_target ON candidates(target_entity_id);
            CREATE TABLE IF NOT EXISTS truth(
              source1_entity_id TEXT NOT NULL, target_entity_id TEXT NOT NULL,
              PRIMARY KEY(source1_entity_id,target_entity_id)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS features(
              source1_entity_id TEXT NOT NULL, target_entity_id TEXT NOT NULL, label INTEGER,
              """ + ",".join(f'"{col}" REAL NOT NULL' for col in FeatureExtractor.feature_columns()) + """,
              PRIMARY KEY(source1_entity_id,target_entity_id)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS val_scores(
              source1_entity_id TEXT NOT NULL,target_entity_id TEXT NOT NULL,label INTEGER NOT NULL,
              probability REAL NOT NULL,PRIMARY KEY(source1_entity_id,target_entity_id)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS predictions(
              source1_entity_id TEXT NOT NULL,target_entity_id TEXT NOT NULL,probability REAL NOT NULL,
              PRIMARY KEY(source1_entity_id,target_entity_id)
            ) WITHOUT ROWID;
        """)

    @staticmethod
    def _signature(path: Path, source_id: int, limit: int | None = None) -> str:
        stat = path.stat()
        return f"{path.resolve()}|{source_id}|{stat.st_size}|{stat.st_mtime_ns}|{limit or 'all'}"

    def _blocking_keys(self, name: str, address: str, country: str) -> list[tuple[int, str]]:
        c = country.casefold().strip()
        n = name.casefold().strip()
        a = address.casefold().strip()
        name_tokens = set(WORD_RE.findall(n))
        informative_name = sorted((w for w in name_tokens if len(w) >= 3 and w.upper() not in STOP),
                                  key=lambda w: (-len(w), w))
        address_tokens = set(WORD_RE.findall(a))
        informative_address = sorted((w for w in address_tokens if len(w) >= 3 and w.upper() not in STOP),
                                     key=lambda w: (-len(w), w))
        result: set[tuple[int, str]] = set()
        if len(n) >= 3:
            result.add((1, f"{c}|{n}"))
        for token in informative_name[:3]:
            result.add((2, f"{c}|{token}"))
        for width in (3, 5, 7):
            if len(n) >= width:
                result.add((3, f"{c}|{n[:width]}"))
        number = NUMBER_RE.search(a)
        if number:
            for token in informative_name[:2]:
                result.add((4, f"{c}|{number.group()}|{token}"))
        if len(a) >= 8:
            result.add((5, f"{c}|{a}"))
        for token in informative_address[:2]:
            result.add((6, f"{c}|{token}"))
        return list(result)

    def _ingest_file(self, db: sqlite3.Connection, split: str, path_value: str, source_id: int,
                     max_rows: int | None = None) -> None:
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset does not exist: {path}")
        sig = self._signature(path, source_id, max_rows)
        state_key = f"data:{source_id}"
        current = db.execute("SELECT signature FROM stage_state WHERE stage=?", (state_key,)).fetchone()
        if current and current[0] == sig:
            print(f"[{split}] source {source_id} already ingested; cache reused")
            return
        progress_key = f"progress:{source_id}"
        progress = db.execute("SELECT signature FROM stage_state WHERE stage=?", (progress_key,)).fetchone()
        checkpoint = 0
        if progress and progress[0].startswith(sig + "|"):
            try:
                checkpoint = int(progress[0].rsplit("|", 1)[1])
                actual = db.execute("SELECT COUNT(*) FROM records WHERE source_id=?", (source_id,)).fetchone()[0]
                if actual != checkpoint:
                    checkpoint = 0
            except ValueError:
                checkpoint = 0
        if checkpoint == 0:
            db.execute("DELETE FROM block_keys WHERE entity_id IN (SELECT entity_id FROM records WHERE source_id=?)", (source_id,))
            db.execute("DELETE FROM records WHERE source_id=?", (source_id,))
            db.execute("DELETE FROM stage_state WHERE stage IN (?,?)", (state_key, progress_key))
            db.commit()
        else:
            print(f"[{split}] resuming source {source_id} after {checkpoint:,} committed rows")
        count = checkpoint
        chunk_size = min(self.chunk_size, max_rows) if max_rows else self.chunk_size
        skipped = 0
        for frame in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                                 usecols=["entity_id", "business_name", "business_address", "country"],
                                 chunksize=chunk_size, encoding="utf-8"):
            if skipped < checkpoint:
                to_skip = min(len(frame), checkpoint - skipped)
                skipped += to_skip
                if to_skip == len(frame):
                    continue
                frame = frame.iloc[to_skip:]
            if max_rows is not None and count + len(frame) > max_rows:
                frame = frame.iloc[:max_rows-count]
            clean = self.cleaner.preprocess_dataset(frame)
            recs, blocks = [], []
            side = 1 if source_id == 1 else 2
            for eid, name, address, country in zip(clean.entity_id, clean.clean_name, clean.clean_address,
                                                    clean.clean_country, strict=True):
                if source_id == 1 and (not eid.startswith("S1-") or not eid[3:].isdigit()):
                    raise ValueError(f"Unexpected Source 1 entity ID format: {eid}")
                if source_id in (2, 3) and not eid.startswith(f"S{source_id}-"):
                    raise ValueError(f"Unexpected source {source_id} entity ID format: {eid}")
                recs.append((eid, source_id, name, address, country))
                blocks.extend((kind, key, side, eid) for kind, key in self._blocking_keys(name, address, country))
            with db:
                db.executemany("INSERT INTO records VALUES(?,?,?,?,?)", recs)
                db.executemany("INSERT OR IGNORE INTO block_keys VALUES(?,?,?,?)", blocks)
            count += len(recs)
            db.execute("INSERT OR REPLACE INTO stage_state VALUES(?,?)", (progress_key, f"{sig}|{count}"))
            db.commit()
            if max_rows is not None and count >= max_rows:
                break
            if count % (self.chunk_size * 10) == 0:
                print(f"[{split}] ingested source {source_id}: {count:,}; RSS={_rss_mb():,.0f} MiB")
            del frame, clean, recs, blocks
        db.execute("INSERT OR REPLACE INTO stage_state VALUES(?,?)", (state_key, sig))
        db.execute("DELETE FROM stage_state WHERE stage=?", (progress_key,))
        db.commit()
        print(f"[{split}] ingested source {source_id}: {count:,} rows")

    def prepare_data(self, split: str, *, include_truth: bool) -> sqlite3.Connection:
        db = self._connect(split)
        key_prefix = "train" if split in ("train", "benchmark") else "test"
        limit = int(self.config.get("benchmark", {}).get("rows_per_source", 50_000)) if split == "benchmark" else None
        self._ingest_file(db, split, self.paths[f"{key_prefix}_source1"], 1, limit)
        self._ingest_file(db, split, self.paths[f"{key_prefix}_source2"], 2, limit)
        self._ingest_file(db, split, self.paths[f"{key_prefix}_source3"], 3, limit)
        if include_truth:
            self._ingest_truth(db, Path(self.paths["train_ground_truth"]))
        return db

    def _ingest_truth(self, db: sqlite3.Connection, path: Path) -> None:
        sig = self._signature(path, 0)
        current = db.execute("SELECT signature FROM stage_state WHERE stage='truth'").fetchone()
        if current and current[0] == sig:
            print("[train] ground truth cache reused")
            return
        db.execute("DELETE FROM truth")
        rows = 0
        for frame in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                                 usecols=["source1_entity_id", "matched_entity_ids"], chunksize=25_000,
                                 encoding="utf-8"):
            pairs = []
            for s1, targets in zip(frame.source1_entity_id, frame.matched_entity_ids, strict=True):
                seen = set()
                for target in targets.split(","):
                    target = target.strip()
                    if target:
                        if not target.startswith(("S2-", "S3-")):
                            raise ValueError(f"Invalid ground-truth target ID: {target}")
                        seen.add(target)
                pairs.extend((s1, target) for target in seen)
            with db:
                db.executemany("INSERT OR IGNORE INTO truth VALUES(?,?)", pairs)
            rows += len(frame)
            del pairs, frame
        db.execute("INSERT OR REPLACE INTO stage_state VALUES('truth',?)", (sig,))
        db.commit()
        print(f"[train] parsed {rows:,} ground-truth S1 rows; {db.execute('SELECT COUNT(*) FROM truth').fetchone()[0]:,} true pairs")

    def generate_candidates(self, db: sqlite3.Connection, split: str) -> dict:
        state = db.execute("SELECT signature FROM stage_state WHERE stage='candidates'").fetchone()
        source_sigs = "|".join(row[0] for row in db.execute("SELECT signature FROM stage_state WHERE stage LIKE 'data:%' ORDER BY stage"))
        signature = f"v5|group={self.max_group_size}|{source_sigs}"
        if not self.force_rebuild and state and state[0] == signature:
            n = db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            print(f"[{split}] candidate cache reused: {n:,} pairs")
            return self._blocking_summary(db)
        db.execute("DELETE FROM candidates")
        db.execute("DROP TABLE IF EXISTS safe_blocks")
        db.execute("CREATE TABLE safe_blocks(kind INTEGER,block_key TEXT,ref_count INTEGER,target_count INTEGER,PRIMARY KEY(kind,block_key)) WITHOUT ROWID")
        db.execute("""INSERT INTO safe_blocks
          SELECT kind,block_key,SUM(CASE WHEN side=1 THEN 1 ELSE 0 END),SUM(CASE WHEN side=2 THEN 1 ELSE 0 END)
          FROM block_keys GROUP BY kind,block_key
          HAVING SUM(CASE WHEN side=1 THEN 1 ELSE 0 END)>0
             AND SUM(CASE WHEN side=2 THEN 1 ELSE 0 END)>0
             AND SUM(CASE WHEN side=1 THEN 1 ELSE 0 END)<=?
             AND SUM(CASE WHEN side=2 THEN 1 ELSE 0 END)<=?""", (self.max_group_size, self.max_group_size))
        db.execute("DROP TABLE IF EXISTS selected_ref_blocks")
        db.execute("CREATE TABLE selected_ref_blocks(kind INTEGER,block_key TEXT,entity_id TEXT,PRIMARY KEY(kind,block_key,entity_id)) WITHOUT ROWID")
        # Select the rarest safe tokens for each S1 using observed cross-source
        # frequencies. This constrains retrieval before pair expansion.
        db.execute("""INSERT INTO selected_ref_blocks
          SELECT kind,block_key,entity_id FROM (
            SELECT b.kind,b.block_key,b.entity_id,
              ROW_NUMBER() OVER(PARTITION BY b.entity_id,b.kind ORDER BY s.target_count,s.ref_count,LENGTH(b.block_key),b.block_key) rank
            FROM block_keys b JOIN safe_blocks s ON s.kind=b.kind AND s.block_key=b.block_key
            WHERE b.side=1 AND b.kind IN (2,6)
          ) WHERE rank <= CASE kind WHEN 2 THEN 2 ELSE 1 END""")
        db.execute("""INSERT OR IGNORE INTO selected_ref_blocks
          SELECT b.kind,b.block_key,b.entity_id FROM block_keys b JOIN safe_blocks s
          ON s.kind=b.kind AND s.block_key=b.block_key WHERE b.side=1 AND b.kind NOT IN (2,6)""")
        db.commit()
        summary = {}
        for kind, name in enumerate(BLOCK_TYPES, 1):
            bit = BLOCK_BITS[name]
            db.execute("""INSERT INTO candidates(source1_entity_id,target_entity_id,block_flags)
                SELECT DISTINCT r.entity_id,t.entity_id,?
                FROM selected_ref_blocks r JOIN safe_blocks s ON s.kind=r.kind AND s.block_key=r.block_key
                JOIN block_keys t ON t.kind=r.kind AND t.block_key=r.block_key
                WHERE r.kind=? AND t.side=2
                ON CONFLICT(source1_entity_id,target_entity_id)
                DO UPDATE SET block_flags=candidates.block_flags | excluded.block_flags""", (bit, kind))
            db.commit()
            group_stats = db.execute("SELECT COUNT(*),COALESCE(SUM(s.ref_count*s.target_count),0) FROM safe_blocks s WHERE kind=? AND EXISTS(SELECT 1 FROM selected_ref_blocks r WHERE r.kind=s.kind AND r.block_key=s.block_key)", (kind,)).fetchone()
            summary[name] = {"safe_groups": int(group_stats[0]), "theoretical_pairs": int(group_stats[1]),
                             "candidate_pairs_after_rule": int(db.execute("SELECT COUNT(*) FROM candidates WHERE (block_flags & ?) != 0", (bit,)).fetchone()[0])}
            print(f"[{split}] block {name}: safe groups={group_stats[0]:,}, theoretical pairs={group_stats[1]:,}, unique candidates with rule={summary[name]['candidate_pairs_after_rule']:,}; RSS={_rss_mb():,.0f} MiB")
        n = db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        db.execute("INSERT OR REPLACE INTO stage_state VALUES('candidates',?)", (signature,))
        db.commit()
        print(f"[{split}] total unique candidate pairs={n:,}")
        return summary

    @staticmethod
    def _blocking_summary(db: sqlite3.Connection) -> dict:
        return {"cached": {"candidate_pairs": db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]}}

    def candidate_recall(self, db: sqlite3.Connection, block_summary: dict) -> dict:
        truth_count = db.execute("SELECT COUNT(*) FROM truth").fetchone()[0]
        found = db.execute("SELECT COUNT(*) FROM truth g JOIN candidates c USING(source1_entity_id,target_entity_id)").fetchone()[0]
        source_recall = {}
        for prefix in ("S2-%", "S3-%"):
            total = db.execute("SELECT COUNT(*) FROM truth WHERE target_entity_id LIKE ?", (prefix,)).fetchone()[0]
            retrieved = db.execute("SELECT COUNT(*) FROM truth g JOIN candidates c USING(source1_entity_id,target_entity_id) WHERE g.target_entity_id LIKE ?", (prefix,)).fetchone()[0]
            source_recall[prefix[:2]] = retrieved / max(1, total)
        country_recall = {}
        for country, total, hit in db.execute("""SELECT r.clean_country,COUNT(*),SUM(CASE WHEN c.target_entity_id IS NULL THEN 0 ELSE 1 END)
             FROM truth g JOIN records r ON r.entity_id=g.source1_entity_id
             LEFT JOIN candidates c USING(source1_entity_id,target_entity_id)
             GROUP BY r.clean_country"""):
            country_recall[country] = hit / max(1, total)
        entity_stats = db.execute("""SELECT COUNT(*),SUM(CASE WHEN COALESCE(h.n,0)=g.n THEN 1 ELSE 0 END),
             SUM(CASE WHEN COALESCE(h.n,0)=0 THEN 1 ELSE 0 END)
             FROM (SELECT source1_entity_id,COUNT(*) n FROM truth GROUP BY source1_entity_id) g
             LEFT JOIN (SELECT source1_entity_id,COUNT(*) n FROM truth t JOIN candidates c USING(source1_entity_id,target_entity_id) GROUP BY source1_entity_id) h USING(source1_entity_id)""").fetchone()
        total_candidates = db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        false_pairs = total_candidates - found
        n_ref = db.execute("SELECT COUNT(*) FROM records WHERE source_id=1").fetchone()[0]
        n_with_candidate = db.execute("SELECT COUNT(DISTINCT source1_entity_id) FROM candidates").fetchone()[0]
        stats = {"mean": total_candidates / max(1, n_ref), "entities_with_candidates": int(n_with_candidate),
                 "entities_without_candidates": int(n_ref - n_with_candidate)}
        distribution_sql = "SELECT COALESCE(c.n,0) FROM records r LEFT JOIN (SELECT source1_entity_id,COUNT(*) n FROM candidates GROUP BY source1_entity_id) c ON c.source1_entity_id=r.entity_id WHERE r.source_id=1 ORDER BY 1"
        for label, p in (("median", .5), ("p90", .90), ("p95", .95), ("p99", .99)):
            offset = int((n_ref - 1) * p)
            value = db.execute(f"SELECT COALESCE(c.n,0) FROM records r LEFT JOIN (SELECT source1_entity_id,COUNT(*) n FROM candidates GROUP BY source1_entity_id) c ON c.source1_entity_id=r.entity_id WHERE r.source_id=1 ORDER BY 1 LIMIT 1 OFFSET {offset}").fetchone()[0] if n_ref else 0
            stats[label] = int(value)
        stats["max"] = int(db.execute("SELECT COALESCE(MAX(n),0) FROM (SELECT COUNT(*) n FROM candidates GROUP BY source1_entity_id)").fetchone()[0])
        report = {"status": "measured", "true_match_pairs": int(truth_count), "retrieved_true_match_pairs": int(found),
                  "candidate_recall": found / max(1, truth_count), "source_recall": source_recall,
                  "country_recall": country_recall, "positive_match_entities": int(entity_stats[0]),
                  "all_match_entity_coverage": entity_stats[1] / max(1, entity_stats[0]),
                  "zero_retrieved_true_match_entities": int(entity_stats[2]),
                  "all_test_source1_entities": int(n_ref), "candidate_pairs": int(total_candidates),
                  "false_candidate_pairs": int(false_pairs), "candidates_per_entity": stats,
                  "blocking_rules": block_summary}
        out = self.root / "artifacts/reports/candidate_recall_report.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        text = ["Candidate Recall Report", f"True match pairs: {truth_count:,}", f"Retrieved true pairs: {found:,}",
                f"Overall pair recall: {report['candidate_recall']:.4%}",
                f"All-match entity coverage among positive S1: {report['all_match_entity_coverage']:.4%}",
                f"Positive S1 with zero retrieved true matches: {report['zero_retrieved_true_match_entities']:,}",
                f"Candidate pairs: {total_candidates:,}", f"False candidate pairs: {false_pairs:,}",
                f"Candidates/S1: {json.dumps(stats)}", f"S2/S3 recall: {json.dumps(source_recall)}",
                f"Country recall: {json.dumps(country_recall)}"]
        (self.root / "artifacts/reports/candidate_recall_report.txt").write_text("\n".join(text)+"\n", encoding="utf-8")
        print(f"Candidate recall={report['candidate_recall']:.4%}; all-match entity coverage={report['all_match_entity_coverage']:.4%}; pairs={total_candidates:,}")
        return report

    def _feature_schema(self, db: sqlite3.Connection, include_label: bool) -> None:
        version = hashlib.sha256(("feature-v5|" + "|".join(self.feature_cols)).encode()).hexdigest()
        current = db.execute("SELECT signature FROM stage_state WHERE stage='feature_schema'").fetchone()
        if current and current[0] != version:
            db.execute("DROP TABLE IF EXISTS features")
        if not current or current[0] != version:
            definitions = ",".join(f'"{column}" REAL NOT NULL' for column in self.feature_cols)
            db.execute(f"CREATE TABLE IF NOT EXISTS features(source1_entity_id TEXT NOT NULL,target_entity_id TEXT NOT NULL,label INTEGER,{definitions},PRIMARY KEY(source1_entity_id,target_entity_id)) WITHOUT ROWID")
            db.execute("INSERT OR REPLACE INTO stage_state VALUES('feature_schema',?)", (version,))
            db.commit()

    def build_features(self, db: sqlite3.Connection, split: str, row_limit: int | None = None) -> int:
        started = time.perf_counter()
        rss_start = _rss_mb()
        self._feature_schema(db, split == "train")
        total = db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        done = db.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        if done == total and not self.force_rebuild:
            print(f"[{split}] feature cache reused: {done:,} rows")
            if row_limit is None:
                self._write_feature_report(split, total, 0, started, rss_start, reused=True)
            return done
        if self.force_rebuild:
            db.execute("DELETE FROM features")
            db.commit()
        reader = sqlite3.connect(self.db_paths[split], timeout=120)
        reader.execute("PRAGMA query_only=ON")
        label_sql = "CASE WHEN g.source1_entity_id IS NULL THEN 0 ELSE 1 END" if split == "train" else "NULL"
        query = f"""SELECT c.source1_entity_id,c.target_entity_id,c.block_flags,
                 r.clean_name,r.clean_address,r.clean_country,t.clean_name,t.clean_address,t.clean_country,{label_sql}
          FROM candidates c JOIN records r ON r.entity_id=c.source1_entity_id
          JOIN records t ON t.entity_id=c.target_entity_id
          LEFT JOIN truth g ON g.source1_entity_id=c.source1_entity_id AND g.target_entity_id=c.target_entity_id
          WHERE NOT EXISTS(SELECT 1 FROM features f WHERE f.source1_entity_id=c.source1_entity_id AND f.target_entity_id=c.target_entity_id)
          ORDER BY c.source1_entity_id,c.target_entity_id""" + (f" LIMIT {int(row_limit)}" if row_limit else "")
        cursor = reader.execute(query)
        insert_cols = ["source1_entity_id", "target_entity_id", "label", *self.feature_cols]
        placeholders = ",".join("?" for _ in insert_cols)
        sql_insert = f"INSERT OR REPLACE INTO features({','.join(insert_cols)}) VALUES({placeholders})"
        processed = done
        while True:
            rows = cursor.fetchmany(self.feature_chunk_size)
            if not rows:
                break
            joined = pd.DataFrame.from_records(rows, columns=["source1_entity_id", "target_entity_id", "block_flags",
                "ref_name", "ref_address", "ref_country", "target_name", "target_address", "target_country", "label"])
            pairs = joined[["source1_entity_id", "target_entity_id", "block_flags"]]
            refs = joined[["source1_entity_id", "ref_name", "ref_address", "ref_country"]].drop_duplicates("source1_entity_id")
            refs.columns = ["entity_id", "clean_name", "clean_address", "clean_country"]
            targets = joined[["target_entity_id", "target_name", "target_address", "target_country"]].drop_duplicates("target_entity_id")
            targets.columns = ["entity_id", "clean_name", "clean_address", "clean_country"]
            extracted = self.extractor.generate_candidate_features(pairs, refs, targets)
            extracted.insert(2, "label", joined.label.to_numpy())
            data = extracted[insert_cols].replace([np.inf, -np.inf], 0).fillna(0)
            values = data.itertuples(index=False, name=None)
            with db:
                db.executemany(sql_insert, values)
            processed += len(rows)
            if processed % (self.feature_chunk_size * 20) < len(rows):
                print(f"[{split}] featured {processed:,}/{total:,}; RSS={_rss_mb():,.0f} MiB; D free={shutil.disk_usage(self.root).free/1024**3:.1f} GiB")
            del rows, joined, pairs, refs, targets, extracted, data
        reader.close()
        if row_limit is None and processed != total:
            raise RuntimeError(f"Feature stage wrote {processed:,} of {total:,} candidate rows")
        print(f"[{split}] feature stage complete: {processed:,} rows, {len(self.feature_cols)} numeric features")
        if row_limit is None:
            self._write_feature_report(split, total, processed - done, started, rss_start, reused=False)
        return processed

    def _write_feature_report(self, split: str, candidate_count: int, new_rows: int,
                              started: float, rss_start: float, *, reused: bool) -> None:
        report = {
            "status": "measured", "split": split, "candidate_pairs": int(candidate_count),
            "feature_rows_persisted": int(candidate_count), "new_rows_this_run": int(new_rows),
            "feature_count": len(self.feature_cols), "feature_columns": self.feature_cols,
            "storage": "SQLite numeric REAL columns", "chunk_size": self.feature_chunk_size,
            "elapsed_seconds": time.perf_counter() - started, "rss_start_mib": rss_start,
            "rss_end_mib": _rss_mb(), "cache_reused": reused,
            "feature_schema_sha256": hashlib.sha256(("feature-v5|" + "|".join(self.feature_cols)).encode()).hexdigest(),
        }
        path = self.root / "artifacts/reports" / f"feature_report_{split}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if split == "train":
            (self.root / "artifacts/reports/feature_report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")

    def _split_sql(self, split_value: int, entity_column: str = "source1_entity_id") -> str:
        # Numeric S1 suffixes are challenge-defined. This deterministic modulo
        # split operates on entities, never individual candidate rows.
        if entity_column not in {"source1_entity_id", "entity_id"}:
            raise ValueError(f"Unsupported entity column for split: {entity_column}")
        return f"CAST(SUBSTR({entity_column},4) AS INTEGER) % 5 = {split_value}"

    class _SqlIter(xgb.DataIter):
        def __init__(self, db_path: Path, query: str, feature_cols: list[str], batch_size: int,
                     cache_prefix: Path):
            self.db_path, self.query, self.feature_cols, self.batch_size = db_path, query, feature_cols, batch_size
            self.reader = None
            super().__init__(cache_prefix=str(cache_prefix), release_data=True, on_host=True)

        def reset(self):
            if self.reader:
                self.reader.close()
            self.reader = sqlite3.connect(self.db_path)
            self.reader.execute("PRAGMA query_only=ON")
            self.cursor = self.reader.execute(self.query)

        def next(self, input_data):
            rows = self.cursor.fetchmany(self.batch_size)
            if not rows:
                self.reader.close()
                self.reader = None
                return 0
            matrix = np.asarray([row[1:] for row in rows], dtype=np.float32)
            labels = np.asarray([row[0] for row in rows], dtype=np.float32)
            input_data(data=matrix, label=labels, feature_names=self.feature_cols)
            return 1

    def _ext_matrix(self, db: sqlite3.Connection, split: str, validation: bool, ref=None):
        where = self._split_sql(0) if validation else f"NOT ({self._split_sql(0)})"
        query = f"SELECT label,{','.join(chr(34)+c+chr(34) for c in self.feature_cols)} FROM features WHERE {where} AND label IS NOT NULL ORDER BY source1_entity_id,target_entity_id"
        iterator = self._SqlIter(self.db_paths[split], query, self.feature_cols, self.feature_chunk_size,
                                 self.temp / f"xgb_{split}_{'val' if validation else 'train'}")
        return xgb.ExtMemQuantileDMatrix(iterator, max_bin=256, nthread=4, ref=ref)

    def _score_validation(self, db: sqlite3.Connection, booster: xgb.Booster, split: str) -> tuple[float, dict]:
        db.execute("DELETE FROM val_scores")
        where = self._split_sql(0)
        query = f"SELECT source1_entity_id,target_entity_id,label,{','.join(chr(34)+c+chr(34) for c in self.feature_cols)} FROM features WHERE {where} AND label IS NOT NULL ORDER BY source1_entity_id,target_entity_id"
        reader = sqlite3.connect(self.db_paths[split])
        cursor = reader.execute(query)
        while True:
            rows = cursor.fetchmany(self.feature_chunk_size)
            if not rows:
                break
            ids = [(r[0], r[1], int(r[2])) for r in rows]
            matrix = np.asarray([r[3:] for r in rows], dtype=np.float32)
            dm = xgb.DMatrix(matrix, feature_names=self.feature_cols, nthread=4)
            probabilities = booster.predict(dm, iteration_range=(0, getattr(booster, "best_iteration", 0) + 1))
            with db:
                db.executemany("INSERT OR REPLACE INTO val_scores VALUES(?,?,?,?)",
                               ((s1, target, label, float(prob)) for (s1, target, label), prob in zip(ids, probabilities, strict=True)))
            del rows, ids, matrix, dm, probabilities
        reader.close()
        val_entities = db.execute(
            f"SELECT COUNT(*) FROM records WHERE source_id=1 AND {self._split_sql(0, 'entity_id')}"
        ).fetchone()[0]
        n_thresholds = int(self.config["model"].get("validation_thresholds", 25))
        thresholds = np.linspace(.05, .99, n_thresholds)
        report_rows = []
        for threshold in thresholds:
            db.execute("DROP TABLE IF EXISTS temp.pred_agg")
            db.execute("CREATE TEMP TABLE pred_agg AS SELECT source1_entity_id,COUNT(*) predicted,SUM(label) tp FROM val_scores WHERE probability>=? GROUP BY source1_entity_id", (float(threshold),))
            row = db.execute("""WITH truth_agg AS (SELECT source1_entity_id,COUNT(*) actual FROM truth GROUP BY source1_entity_id),
              per_entity AS (SELECT r.entity_id,COALESCE(t.actual,0) actual,COALESCE(p.predicted,0) predicted,COALESCE(p.tp,0) tp
                FROM records r LEFT JOIN truth_agg t ON t.source1_entity_id=r.entity_id
                LEFT JOIN pred_agg p ON p.source1_entity_id=r.entity_id
                WHERE r.source_id=1 AND CAST(SUBSTR(r.entity_id,4) AS INTEGER)%5=0)
              SELECT AVG(CASE WHEN actual=0 AND predicted=0 THEN 1.0 WHEN predicted=0 THEN 0.0 ELSE 1.0*tp/predicted END),
                     AVG(CASE WHEN actual=0 AND predicted=0 THEN 1.0 WHEN actual=0 THEN 0.0 ELSE 1.0*tp/actual END),
                     AVG(CASE WHEN actual=0 AND predicted=0 THEN 1.0 WHEN tp=0 THEN 0.0 ELSE 1.25*(1.0*tp/predicted)*(1.0*tp/actual)/(0.25*(1.0*tp/predicted)+(1.0*tp/actual)) END),
                     SUM(predicted),SUM(CASE WHEN predicted=0 THEN 1 ELSE 0 END),SUM(CASE WHEN predicted>0 THEN 1 ELSE 0 END)
              FROM per_entity""").fetchone()
            report_rows.append({"threshold": float(threshold), "macro_precision": float(row[0] or 0),
                "macro_recall": float(row[1] or 0), "macro_f0.5": float(row[2] or 0),
                "predicted_match_count": int(row[3] or 0), "zero_match_entity_count": int(row[4] or 0),
                "entities_with_matches": int(row[5] or 0), "validation_entities": int(val_entities)})
        best = max(report_rows, key=lambda item: (item["macro_f0.5"], item["macro_precision"], item["threshold"]))
        report_path = self.root / "artifacts/reports/threshold_report.csv"
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(report_rows[0]))
            writer.writeheader(); writer.writerows(report_rows)
        # Retain the previous path for downstream integrations while publishing
        # the concise Stage 4 report name requested for handoff.
        shutil.copy2(report_path, self.root / "artifacts/reports/threshold_sweep.csv")
        (self.root / "artifacts/reports/validation_metrics.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        self._error_analysis(db, best["threshold"])
        (self.root / "artifacts/reports/validation_report.json").write_text(
            json.dumps({**best, "metric": "source1_entity_macro_f0.5",
                        "split": "source1_suffix_modulo_5_equals_0",
                        "threshold_grid_size": n_thresholds,
                        "error_analysis": str(self.root / "artifacts/reports/validation_error_analysis.json")}, indent=2),
            encoding="utf-8")
        return best["threshold"], best

    def _error_analysis(self, db: sqlite3.Connection, threshold: float) -> None:
        db.execute("DROP TABLE IF EXISTS temp.pred_agg")
        db.execute("CREATE TEMP TABLE pred_agg AS SELECT source1_entity_id,COUNT(*) predicted,SUM(label) tp FROM val_scores WHERE probability>=? GROUP BY source1_entity_id", (threshold,))
        row = db.execute("""SELECT
          SUM(CASE WHEN probability>=? AND label=0 THEN 1 ELSE 0 END),
          SUM(CASE WHEN probability<? AND label=1 THEN 1 ELSE 0 END),
          (SELECT COUNT(*) FROM truth g LEFT JOIN candidates c USING(source1_entity_id,target_entity_id)
             JOIN records r ON r.entity_id=g.source1_entity_id WHERE c.source1_entity_id IS NULL AND CAST(SUBSTR(g.source1_entity_id,4) AS INTEGER)%5=0),
          (SELECT COUNT(*) FROM records r LEFT JOIN (SELECT source1_entity_id,COUNT(*) actual FROM truth GROUP BY source1_entity_id) t ON t.source1_entity_id=r.entity_id
             LEFT JOIN pred_agg p ON p.source1_entity_id=r.entity_id WHERE r.source_id=1 AND CAST(SUBSTR(r.entity_id,4) AS INTEGER)%5=0 AND COALESCE(t.actual,0)=0 AND COALESCE(p.predicted,0)>0)
          FROM val_scores WHERE source1_entity_id IN (SELECT entity_id FROM records WHERE source_id=1 AND CAST(SUBSTR(entity_id,4) AS INTEGER)%5=0)""", (threshold, threshold)).fetchone()
        report = {"threshold": threshold, "classification_false_positives": int(row[0] or 0),
                  "classification_false_negatives": int(row[1] or 0), "retrieval_failure_true_pairs": int(row[2] or 0),
                  "false_positive_singleton_entities": int(row[3] or 0)}
        report["validation_entities_with_multiple_true_matches"] = db.execute("""SELECT COUNT(*) FROM
          (SELECT source1_entity_id,COUNT(*) n FROM truth GROUP BY source1_entity_id HAVING n>1) t
          JOIN records r ON r.entity_id=t.source1_entity_id WHERE CAST(SUBSTR(t.source1_entity_id,4) AS INTEGER)%5=0""").fetchone()[0]
        report["validation_entities_with_multiple_predictions"] = db.execute("SELECT COUNT(*) FROM (SELECT source1_entity_id,COUNT(*) n FROM val_scores WHERE probability>=? GROUP BY source1_entity_id HAVING n>1)", (threshold,)).fetchone()[0]
        path = self.root / "artifacts/reports/validation_error_analysis.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        for filename, query, params in (
            ("false_positive_pairs.csv", "SELECT source1_entity_id,target_entity_id,probability FROM val_scores WHERE probability>=? AND label=0 ORDER BY probability DESC LIMIT 1000", (threshold,)),
            ("classification_false_negative_pairs.csv", "SELECT source1_entity_id,target_entity_id,probability FROM val_scores WHERE probability<? AND label=1 ORDER BY probability LIMIT 1000", (threshold,)),
            ("retrieval_failure_pairs.csv", "SELECT g.source1_entity_id,g.target_entity_id FROM truth g LEFT JOIN candidates c USING(source1_entity_id,target_entity_id) WHERE c.source1_entity_id IS NULL AND CAST(SUBSTR(g.source1_entity_id,4) AS INTEGER)%5=0 ORDER BY g.source1_entity_id,g.target_entity_id LIMIT 1000", ()),
        ):
            with (self.root / "artifacts/reports" / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream); writer.writerow(["source1_entity_id", "target_entity_id", "probability"] if "probability" in query else ["source1_entity_id", "target_entity_id"])
                writer.writerows(db.execute(query, params))

    def train_model(self, db: sqlite3.Connection, split: str) -> dict:
        started = time.perf_counter()
        counts = {}
        for label, is_validation in ((0, False), (1, False), (0, True), (1, True)):
            where = self._split_sql(0) if is_validation else f"NOT ({self._split_sql(0)})"
            counts[f"{'validation' if is_validation else 'train'}_{'positive' if label else 'negative'}"] = db.execute(
                f"SELECT COUNT(*) FROM features WHERE label=? AND {where}", (label,)).fetchone()[0]
        if not counts["train_positive"] or not counts["train_negative"] or not counts["validation_positive"]:
            raise ValueError(f"Insufficient train/validation classes for model fit: {counts}")
        print(f"Training class counts: {counts}; RSS={_rss_mb():,.0f} MiB")
        dtrain = self._ext_matrix(db, split, False)
        dval = self._ext_matrix(db, split, True, ref=dtrain)
        params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
                  "max_depth": 7, "eta": .05, "min_child_weight": 2, "subsample": .85,
                  "colsample_bytree": .85, "reg_lambda": 2, "nthread": 4,
                  "seed": int(self.config["model"].get("random_state", 42)),
                  "scale_pos_weight": counts["train_negative"] / max(1, counts["train_positive"])}
        evals_result = {}
        val_booster = xgb.train(params, dtrain, num_boost_round=400, evals=[(dval, "validation")],
                                early_stopping_rounds=30, evals_result=evals_result, verbose_eval=25)
        threshold, metrics = self._score_validation(db, val_booster, split)
        best_rounds = max(1, int(getattr(val_booster, "best_iteration", 399)) + 1)
        del dtrain, dval, val_booster
        # Deployment fit includes all training entities after threshold selection.
        full_iter = self._SqlIter(self.db_paths[split],
            f"SELECT label,{','.join(chr(34)+c+chr(34) for c in self.feature_cols)} FROM features WHERE label IS NOT NULL ORDER BY source1_entity_id,target_entity_id",
            self.feature_cols, self.feature_chunk_size, self.temp / "xgb_train_full")
        dfull = xgb.ExtMemQuantileDMatrix(full_iter, max_bin=256, nthread=4)
        final_model = xgb.train(params, dfull, num_boost_round=best_rounds, verbose_eval=False)
        model_path = Path(self.config["model"]["checkpoint_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        final_model.save_model(model_path)
        gain_by_feature = final_model.get_score(importance_type="gain")
        feature_importance = sorted(((column, float(gain_by_feature.get(column, 0.0)))
                                     for column in self.feature_cols), key=lambda row: row[1], reverse=True)
        with (self.root / "artifacts/reports/feature_importance.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream); writer.writerow(["feature", "gain"]); writer.writerows(feature_importance)
        metadata = {"feature_columns": self.feature_cols, "threshold": threshold, "best_rounds": best_rounds,
                    "training_counts": counts, "model": str(model_path), "xgboost_version": xgb.__version__}
        meta_path = model_path.with_suffix(".metadata.json")
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (self.root / "artifacts/reports/training_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
        training_report = {
            "status": "measured", "model": "XGBoost histogram booster",
            "training_positive_pairs": int(counts["train_positive"]),
            "training_negative_pairs": int(counts["train_negative"]),
            "validation_positive_pairs": int(counts["validation_positive"]),
            "validation_negative_pairs": int(counts["validation_negative"]),
            "training_positive_ratio": counts["train_positive"] / max(1, counts["train_positive"] + counts["train_negative"]),
            "scale_pos_weight": params["scale_pos_weight"], "best_rounds": best_rounds,
            "threshold": threshold, "validation_metrics": metrics,
            "feature_count": len(self.feature_cols), "feature_columns": self.feature_cols,
            "model_path": str(model_path), "threshold_metadata_path": str(meta_path),
            "elapsed_seconds": time.perf_counter() - started,
        }
        (self.root / "artifacts/reports/training_report.json").write_text(
            json.dumps(training_report, indent=2), encoding="utf-8")
        shutil.rmtree(self.temp / "xgb_train", ignore_errors=True)
        shutil.rmtree(self.temp / "xgb_val", ignore_errors=True)
        shutil.rmtree(self.temp / "xgb_train_full", ignore_errors=True)
        print(f"Selected threshold {threshold:.3f}; macro F0.5={metrics['macro_f0.5']:.4f}; final model={model_path}")
        return {"threshold": threshold, **metrics, "training_counts": counts}

    def score_test(self, db: sqlite3.Connection, split: str) -> dict:
        model_path = Path(self.config["model"]["checkpoint_path"])
        metadata = json.loads(model_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        if metadata["feature_columns"] != self.feature_cols:
            raise ValueError("Saved model feature schema differs from current feature extractor")
        model = xgb.Booster(); model.load_model(model_path)
        db.execute("DELETE FROM predictions")
        query = f"SELECT source1_entity_id,target_entity_id,{','.join(chr(34)+c+chr(34) for c in self.feature_cols)} FROM features ORDER BY source1_entity_id,target_entity_id"
        reader = sqlite3.connect(self.db_paths[split]); cursor = reader.execute(query)
        count = 0
        while True:
            rows = cursor.fetchmany(self.feature_chunk_size)
            if not rows: break
            keys = [(r[0], r[1]) for r in rows]
            matrix = np.asarray([r[2:] for r in rows], dtype=np.float32)
            dm = xgb.DMatrix(matrix, feature_names=self.feature_cols, nthread=4)
            probs = model.predict(dm)
            threshold = float(metadata["threshold"])
            with db:
                db.executemany("INSERT OR REPLACE INTO predictions VALUES(?,?,?)",
                    ((s1, target, float(prob)) for (s1, target), prob in zip(keys, probs, strict=True) if prob >= threshold))
            count += len(rows)
            if count % (self.feature_chunk_size * 20) < len(rows):
                print(f"[test] scored {count:,}; RSS={_rss_mb():,.0f} MiB")
            del rows, keys, matrix, dm, probs
        reader.close()
        matches = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        zero = db.execute("SELECT COUNT(*) FROM records r WHERE r.source_id=1 AND NOT EXISTS(SELECT 1 FROM predictions p WHERE p.source1_entity_id=r.entity_id)").fetchone()[0]
        multi = db.execute("SELECT COUNT(*) FROM (SELECT source1_entity_id FROM predictions GROUP BY source1_entity_id HAVING COUNT(*)>1)").fetchone()[0]
        report = {"test_candidate_count": int(count), "predicted_matches": int(matches),
                  "zero_match_source1": int(zero), "multi_match_source1": int(multi),
                  "threshold": float(metadata["threshold"])}
        (self.root / "artifacts/reports/test_prediction_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (self.root / "artifacts/reports/prediction_report.json").write_text(
            json.dumps({**report, "model_path": str(model_path),
                        "output_dir": str(self.root / "submission/output"),
                        "scores_persisted": "selected pairs only"}, indent=2), encoding="utf-8")
        return report

    @staticmethod
    def _write_grouped_ids(db: sqlite3.Connection, ref_query: str, pairs_query: str,
                           output: Path, value_column: str) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        refs = db.execute(ref_query)
        pairs = db.execute(pairs_query)
        current = next(pairs, None)
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", value_column])
            for (s1,) in refs:
                values = []
                while current is not None and current[0] < s1:
                    current = next(pairs, None)
                while current is not None and current[0] == s1:
                    values.append(current[1])
                    current = next(pairs, None)
                writer.writerow([s1, ",".join(values)])

    def write_outputs(self, db: sqlite3.Connection) -> tuple[Path, Path]:
        out = self.root / "submission/output"
        candidate = out / "candidate_pairs.tsv"
        matching = out / "matching_results.tsv"
        refs = "SELECT entity_id FROM records WHERE source_id=1 ORDER BY entity_id"
        self._write_grouped_ids(db, refs,
            "SELECT source1_entity_id,target_entity_id FROM candidates ORDER BY source1_entity_id,target_entity_id",
            candidate, "candidate_entity_ids")
        self._write_grouped_ids(db, refs,
            "SELECT source1_entity_id,target_entity_id FROM predictions ORDER BY source1_entity_id,target_entity_id",
            matching, "matched_entity_ids")
        print(f"Outputs written to D: {matching} and {candidate}")
        return matching, candidate

    def run(self, mode: str) -> dict:
        monitor = ResourceMonitor(self.root, mode.replace("-", "_"))
        monitor.start()
        try:
            return self._run_stage(mode)
        finally:
            monitor.stop()

    def run_from_candidates(self, split: str, candidate_db: str | Path) -> dict:
        """Run assigned feature/model/prediction stages against an existing cache.

        This deliberately bypasses ``prepare_data`` and ``generate_candidates``.
        The supplied SQLite database must already contain records and candidates;
        training additionally requires the existing truth table.
        """
        if split not in {"train", "test"}:
            raise ValueError("Candidate-only execution supports split='train' or split='test'")
        path = Path(candidate_db).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Candidate SQLite artifact does not exist: {path}")
        self.db_paths[split] = path
        monitor = ResourceMonitor(self.root, f"{split}_from_candidates")
        started = time.perf_counter()
        monitor.start()
        db = None
        try:
            db = sqlite3.connect(path, timeout=120)
            db.execute("PRAGMA temp_store=FILE")
            tables = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            required = {"records", "candidates", "features", "stage_state"}
            missing = required - tables
            if missing:
                raise ValueError(f"Candidate artifact {path} is missing tables: {sorted(missing)}")
            if split == "train" and "truth" not in tables:
                raise ValueError(f"Training candidate artifact {path} has no truth table")
            candidate_count = int(db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
            source1_count = int(db.execute(
                "SELECT COUNT(*) FROM records WHERE source_id=1").fetchone()[0])
            target_count = int(db.execute(
                "SELECT COUNT(*) FROM records WHERE source_id IN (2,3)").fetchone()[0])
            if candidate_count == 0 or source1_count == 0 or target_count == 0:
                raise ValueError(
                    f"No usable candidate cache in {path}: candidates={candidate_count}, "
                    f"source1={source1_count}, targets={target_count}")
            if split == "train":
                truth_count = int(db.execute("SELECT COUNT(*) FROM truth").fetchone()[0])
                if truth_count == 0:
                    raise ValueError(f"Training candidate artifact {path} has an empty truth table")
            else:
                truth_count = None

            # Candidate-only mode must not silently discard an existing feature
            # cache when the schema version differs. The regular rebuild path can
            # intentionally recreate features; this handoff path is conservative.
            schema_version = hashlib.sha256(
                ("feature-v5|" + "|".join(self.feature_cols)).encode()).hexdigest()
            feature_state = db.execute(
                "SELECT signature FROM stage_state WHERE stage='feature_schema'").fetchone()
            existing_features = int(db.execute("SELECT COUNT(*) FROM features").fetchone()[0])
            if existing_features and (not feature_state or feature_state[0] != schema_version):
                raise ValueError(
                    f"Candidate artifact has {existing_features:,} cached features with a "
                    "different or unknown schema marker; refusing to replace them in candidate-only mode")

            feature_started = time.perf_counter()
            feature_count = self.build_features(db, split)
            feature_seconds = time.perf_counter() - feature_started
            result: dict[str, Any]
            if split == "train":
                result = self.train_model(db, split)
            else:
                result = self.score_test(db, split)
                self.write_outputs(db)
            lineage = {
                "status": "measured", "mode": f"{split}_from_candidates",
                "candidate_database": str(path), "upstream_ingestion_skipped": True,
                "upstream_candidate_generation_skipped": True,
                "candidate_count": candidate_count, "feature_count_per_pair": len(self.feature_cols),
                "feature_rows": int(feature_count), "feature_seconds": feature_seconds,
                "source1_records": source1_count, "target_records": target_count,
                "truth_pairs": truth_count, "result": result,
                "total_seconds": time.perf_counter() - started,
            }
            report_path = self.root / "artifacts/reports" / f"{split}_from_candidates_report.json"
            report_path.write_text(json.dumps(lineage, indent=2), encoding="utf-8")
            print(f"[{split}] candidate-only run complete; upstream ingestion and blocking skipped; "
                  f"candidates={candidate_count:,}; features={feature_count:,}; "
                  f"report={report_path}")
            return lineage
        finally:
            if db is not None:
                db.close()
            monitor.stop()

    def _run_stage(self, mode: str) -> dict:
        started = time.perf_counter()
        split = "train" if mode in ("train", "full", "candidate-recall") else ("benchmark" if mode == "benchmark" else "test")
        db = self.prepare_data(split, include_truth=split == "train")
        block_summary = self.generate_candidates(db, split)
        if split == "train":
            recall_report = self.candidate_recall(db, block_summary)
            if mode == "candidate-recall":
                db.close()
                print(f"[train] candidate-recall elapsed={time.perf_counter()-started:.1f}s; peak RSS not measured, current RSS={_rss_mb():.0f} MiB")
                return {"stage": "candidate-recall"}
        benchmark_rows = self.config.get("benchmark", {}).get("feature_rows") if split == "benchmark" else None
        self.build_features(db, split, int(benchmark_rows) if benchmark_rows else None)
        if split == "benchmark":
            result = {"benchmark_records": db.execute("SELECT COUNT(*) FROM records").fetchone()[0],
                      "benchmark_candidates": db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0],
                      "benchmark_features": db.execute("SELECT COUNT(*) FROM features").fetchone()[0],
                      "rss_mib": _rss_mb(), "elapsed_seconds": time.perf_counter()-started,
                      "db_path": str(self.db_paths[split])}
            db.close()
            (self.root / "artifacts/reports/benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result
        if split == "train":
            result = self.train_model(db, split)
            experiment_path = self.root / "artifacts/reports/experiment_results.csv"
            exists = experiment_path.is_file()
            with experiment_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["experiment", "candidate_recall", "macro_precision", "macro_recall", "macro_f0.5", "candidate_count", "runtime_seconds", "notes"])
                if not exists: writer.writeheader()
                writer.writerow({"experiment": "sqlite_multiblock_xgboost", "candidate_recall": recall_report["candidate_recall"],
                    "macro_precision": result["macro_precision"], "macro_recall": result["macro_recall"],
                    "macro_f0.5": result["macro_f0.5"], "candidate_count": recall_report["candidate_pairs"],
                    "runtime_seconds": round(time.perf_counter()-started, 1), "notes": "first measured full-data run"})
            db.close()
            print(f"[train] total elapsed={time.perf_counter()-started:.1f}s; current RSS={_rss_mb():.0f} MiB")
            return result
        result = self.score_test(db, split)
        self.write_outputs(db)
        db.close()
        print(f"[test] total elapsed={time.perf_counter()-started:.1f}s; current RSS={_rss_mb():.0f} MiB")
        return result

    def validate(self) -> int:
        output = self.root / "submission/output"
        validator = Path(self.config["challenge"]["validator"])
        if not validator.is_file():
            raise FileNotFoundError(f"Official validator not found: {validator}")
        command = [sys.executable, str(validator), "--matching", str(output / "matching_results.tsv"),
                   "--candidate", str(output / "candidate_pairs.tsv"), "--test-dir", self.config["challenge"]["test_dir"]]
        result = subprocess.run(command, check=False, env=os.environ.copy())
        (self.root / "artifacts/reports/validator_exit_code.txt").write_text(str(result.returncode)+"\n", encoding="utf-8")
        return result.returncode

    def package(self) -> Path:
        base = self.root / "submission"
        package = base / "package"
        code = package / "code/business_entity_resolution"
        if package.exists():
            shutil.rmtree(package)
        code.mkdir(parents=True)
        shutil.copytree(self.project_root / "src", code / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(self.project_root / "main.py", code / "main.py")
        shutil.copy2(self.project_root / "requirements.txt", code / "requirements.txt")
        shutil.copy2(self.project_root / "README.md", code / "README.md")
        shutil.copytree(self.project_root / "config", code / "config", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(self.project_root / "Documentation_template.md", package / "Documentation_template.md")
        shutil.copytree(base / "output", package / "output", dirs_exist_ok=True)
        zip_path = base / "Tech_Giants_submission.zip"
        zip_path.unlink(missing_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for file in package.rglob("*"):
                if file.is_file(): archive.write(file, file.relative_to(package).as_posix())
        with zipfile.ZipFile(zip_path) as archive:
            bad = archive.testzip()
            if bad:
                raise IOError(f"ZIP integrity check failed at {bad}")
            required = {"output/matching_results.tsv", "output/candidate_pairs.tsv",
                        "code/business_entity_resolution/main.py",
                        "code/business_entity_resolution/README.md",
                        "code/business_entity_resolution/requirements.txt",
                        "Documentation_template.md"}
            missing = required - set(archive.namelist())
            if missing:
                raise IOError(f"Submission ZIP is missing required entries: {sorted(missing)}")
        print(f"Submission ZIP created: {zip_path} ({zip_path.stat().st_size:,} bytes)")
        return zip_path
