"""Disk-backed, resumable Person 3 candidate retrieval.

This builder writes the SQLite tables consumed by DiskPipeline's Stage 3–5
candidate-only path. It uses only challenge TSVs and deterministic local keys.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import psutil
from rapidfuzz import fuzz

from src.features.string_metrics import FeatureExtractor
from src.preprocessing.text_cleaner import TextCleaner


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
NUMBER_RE = re.compile(r"(?<!\w)\d{1,8}(?!\w)")
STOP = {
    "THE", "INC", "LLC", "LTD", "LIMITED", "CORP", "CORPORATION", "GROUP",
    "COMPANY", "CO", "GLOBAL", "INTERNATIONAL", "SERVICES", "HOLDINGS",
    "ENTERPRISES", "ASSOCIATES", "SOLUTIONS", "AND", "PTE", "PVT", "PRIVATE",
    "GMBH", "SA", "BV", "SL", "AG", "TECHNOLOGIES", "STREET", "ROAD", "RD",
    "ST", "AVENUE", "AVE", "LANE", "DRIVE", "DR", "NEAR", "OPPOSITE",
}

# Candidate provenance bits are deliberately separate from Person 2's six
# feature provenance bits. Existing `block_flags` remains backward compatible.
RETRIEVAL_FAMILIES: dict[int, tuple[str, int]] = {
    1: ("exact_name", 1),
    2: ("country_exact_name", 1),
    3: ("compact_name", 2),
    4: ("unicode_ascii_name", 2),
    5: ("token_signature", 2),
    7: ("rare_token", 2),
    9: ("address_token", 32),
    10: ("token_prefix_fuzzy", 4),
    11: ("exact_address", 16),
    12: ("country_address_number", 8),
    13: ("name_address_number", 8),
    14: ("rare_token_pair", 2),
    15: ("name_address_token", 32),
    16: ("country_rare_name_token", 2),
    17: ("country_address_token", 32),
    18: ("address_number_token", 8),
}

RETRIEVAL_BITS = {kind: 1 << (kind - 1) for kind in RETRIEVAL_FAMILIES}
SCHEMA_VERSION = "person3-hybrid-v4"


def _hash(value: str) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()


def _ascii(value: str) -> str:
    # A small deterministic transliteration table handles common Latin letters
    # that NFKD alone does not decompose. No network or external data is used.
    value = value.translate(str.maketrans({
        "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O",
        "ß": "ss", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "þ": "th", "Þ": "TH",
    }))
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").casefold()


def _informative_tokens(value: str) -> list[str]:
    return sorted({token.casefold() for token in TOKEN_RE.findall(value)
                   if len(token) >= 3 and token.upper() not in STOP})


class Person3CandidateBuilder:
    """Build one train or test candidate SQLite without large Python collections."""

    def __init__(self, *, root: str | Path, mappings: str | Path,
                 chunk_rows: int = 5_000, max_block_pairs: int = 250_000,
                 name_tokens_per_record: int = 6, address_tokens_per_record: int = 2,
                 rows_per_source: int | None = None, max_s1_id: int | None = None,
                 fuzzy_threshold: int = 82, fuzzy_trigger_candidate_count: int = 2,
                 disabled_families: Iterable[int] = ()):
        self.root = Path(root).resolve()
        self.chunk_rows = int(chunk_rows)
        self.max_block_pairs = int(max_block_pairs)
        self.name_tokens_per_record = int(name_tokens_per_record)
        self.address_tokens_per_record = int(address_tokens_per_record)
        self.rows_per_source = rows_per_source
        self.max_s1_id = max_s1_id
        self.fuzzy_threshold = int(fuzzy_threshold)
        self.fuzzy_trigger_candidate_count = int(fuzzy_trigger_candidate_count)
        self.disabled_families = frozenset(int(kind) for kind in disabled_families)
        unknown_families = self.disabled_families - set(RETRIEVAL_FAMILIES)
        if unknown_families:
            raise ValueError(f"Unknown retrieval family IDs: {sorted(unknown_families)}")
        self.phase_times: dict[str, float] = {}
        if min(self.chunk_rows, self.max_block_pairs, self.name_tokens_per_record,
               self.address_tokens_per_record) <= 0:
            raise ValueError("Chunk, block, and token limits must be positive")
        self.cleaner = TextCleaner(str(Path(mappings).resolve()))
        self.process = psutil.Process()
        self.started = 0.0
        self.phase_started = 0.0
        self.db_path: Path | None = None
        self.input_paths: dict[str, str] = {}
        self.split = ""
        self.signature = ""
        self.rss_observed_peak_mib = 0.0
        self.disk_free_start = 0
        self.disk_free_minimum = 0

    def _print_progress(self, phase: str, rows: int, extra: str = "") -> None:
        elapsed = time.perf_counter() - self.phase_started
        rss = self.process.memory_info().rss / (1024 * 1024)
        free_bytes = shutil.disk_usage(self.root).free
        self.rss_observed_peak_mib = max(self.rss_observed_peak_mib, rss)
        self.disk_free_minimum = min(self.disk_free_minimum, free_bytes)
        free = free_bytes / (1024 ** 3)
        speed = rows / max(elapsed, 1e-6)
        print(f"[{self.split}] {phase}: rows={rows:,}; elapsed={elapsed:.1f}s; "
              f"rate={speed:,.0f}/s; RSS={rss:,.0f} MiB; D_free={free:.1f} GiB {extra}",
              flush=True)

    def _connect(self, path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size:
            # Never migrate or silently adopt an arbitrary pre-existing DB.
            probe = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            try:
                tables = {r[0] for r in probe.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                if "stage_state" not in tables:
                    raise ValueError(f"Refusing to reuse non-Person-3 database: {path}")
                marker = probe.execute(
                    "SELECT signature FROM stage_state WHERE stage='person3_signature'").fetchone()
                if not marker:
                    raise ValueError(f"Refusing to modify unmarked existing SQLite database: {path}")
            finally:
                probe.close()
        db = sqlite3.connect(path, timeout=120)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-262144")
        db.execute("PRAGMA mmap_size=536870912")
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS records(
              entity_id TEXT PRIMARY KEY, source_id INTEGER NOT NULL,
              clean_name TEXT NOT NULL, clean_address TEXT NOT NULL, clean_country TEXT NOT NULL,
              entity_num INTEGER, raw_name TEXT NOT NULL DEFAULT '',
              raw_address TEXT NOT NULL DEFAULT '', raw_country TEXT NOT NULL DEFAULT ''
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS candidates(
              source1_entity_id TEXT NOT NULL, target_entity_id TEXT NOT NULL,
              block_flags INTEGER NOT NULL, retrieval_flags INTEGER NOT NULL DEFAULT 0,
              PRIMARY KEY(source1_entity_id,target_entity_id)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS truth(
              source1_entity_id TEXT NOT NULL, target_entity_id TEXT NOT NULL,
              PRIMARY KEY(source1_entity_id,target_entity_id)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS stage_state(
              stage TEXT PRIMARY KEY, signature TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS retrieval_keys(
              kind INTEGER NOT NULL, key_hash TEXT NOT NULL, key_value TEXT NOT NULL,
              side INTEGER NOT NULL, entity_id TEXT NOT NULL,
              PRIMARY KEY(kind,key_hash,entity_id)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS record_numbers(
              entity_id TEXT NOT NULL, side INTEGER NOT NULL, number TEXT NOT NULL,
              PRIMARY KEY(entity_id,number)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS person3_block_stats(
              kind INTEGER NOT NULL, key_hash TEXT NOT NULL,
              ref_count INTEGER NOT NULL, target_count INTEGER NOT NULL,
              pair_count INTEGER NOT NULL, included INTEGER NOT NULL,
              PRIMARY KEY(kind,key_hash)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS person3_selected_tokens(
              family INTEGER NOT NULL, entity_id TEXT NOT NULL, side INTEGER NOT NULL,
              key_hash TEXT NOT NULL, key_value TEXT NOT NULL, document_frequency INTEGER NOT NULL,
              PRIMARY KEY(family,entity_id,key_hash)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS person3_misses(
              source1_entity_id TEXT NOT NULL, target_entity_id TEXT NOT NULL,
              source_id INTEGER NOT NULL, raw_s1_name TEXT, raw_s1_address TEXT, raw_s1_country TEXT,
              clean_s1_name TEXT, clean_s1_address TEXT, clean_s1_country TEXT,
              raw_target_name TEXT, raw_target_address TEXT, raw_target_country TEXT,
              clean_target_name TEXT, clean_target_address TEXT, clean_target_country TEXT,
              shared_key_families TEXT NOT NULL, miss_reason TEXT NOT NULL,
              PRIMARY KEY(source1_entity_id,target_entity_id)
            ) WITHOUT ROWID;
        """)
        feature_columns = FeatureExtractor.feature_columns()
        definitions = ",".join(f'"{column}" REAL NOT NULL DEFAULT 0' for column in feature_columns)
        db.execute(f"CREATE TABLE IF NOT EXISTS features(source1_entity_id TEXT NOT NULL,target_entity_id TEXT NOT NULL,label INTEGER,{definitions},PRIMARY KEY(source1_entity_id,target_entity_id)) WITHOUT ROWID")
        # Compatibility with databases created by the current Person 2 schema.
        candidate_cols = {r[1] for r in db.execute("PRAGMA table_info(candidates)")}
        if "retrieval_flags" not in candidate_cols:
            db.execute("ALTER TABLE candidates ADD COLUMN retrieval_flags INTEGER NOT NULL DEFAULT 0")
        record_cols = {r[1] for r in db.execute("PRAGMA table_info(records)")}
        for column, ddl in (("entity_num", "INTEGER"), ("raw_name", "TEXT NOT NULL DEFAULT ''"),
                            ("raw_address", "TEXT NOT NULL DEFAULT ''"),
                            ("raw_country", "TEXT NOT NULL DEFAULT ''")):
            if column not in record_cols:
                db.execute(f"ALTER TABLE records ADD COLUMN {column} {ddl}")
        db.commit()
        return db

    @staticmethod
    def _file_signature(paths: dict[str, str], options: dict[str, Any]) -> str:
        values = []
        for key, value in sorted(paths.items()):
            path = Path(value)
            st = path.stat()
            values.append(f"{key}:{path.resolve()}:{st.st_size}:{st.st_mtime_ns}")
        return hashlib.sha256((SCHEMA_VERSION + "|" + "|".join(values) + "|" +
                               json.dumps(options, sort_keys=True)).encode()).hexdigest()

    def _check_signature(self, db: sqlite3.Connection) -> None:
        row = db.execute("SELECT signature FROM stage_state WHERE stage='person3_signature'").fetchone()
        if row and row[0] != self.signature:
            raise ValueError(
                f"Existing Person 3 database {self.db_path} belongs to different inputs/config. "
                "Use a new output directory; existing cache was left unchanged.")
        if not row:
            existing = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            if existing:
                raise ValueError("Refusing to reinterpret a non-empty SQLite DB without a Person 3 signature")
            db.execute("INSERT INTO stage_state VALUES('person3_signature',?)", (self.signature,))
            db.commit()

    def _keys_for_record(self, name: str, address: str, country: str) -> tuple[list[tuple[int, str]], list[str]]:
        keys: set[tuple[int, str]] = set()
        name_cf = name.casefold().strip()
        address_cf = address.casefold().strip()
        if name_cf:
            keys.add((1, name_cf))
            if country:
                keys.add((2, f"{country.casefold()}\x1f{name_cf}"))
            compact = "".join(TOKEN_RE.findall(name_cf))
            if len(compact) >= 3:
                keys.add((3, compact))
            ascii_name = _ascii(name_cf)
            if ascii_name and ascii_name != name_cf:
                keys.add((4, ascii_name))
            tokens = _informative_tokens(name_cf)
            if tokens:
                keys.add((5, "\x1f".join(sorted(tokens))))
                for token in tokens:
                    keys.add((6, token))
                    if len(token) >= 4:
                        keys.add((10, token[:4]))
        else:
            tokens = []
        if address_cf:
            keys.add((11, address_cf))
            for token in _informative_tokens(address_cf):
                keys.add((8, token))
        numbers = sorted(set(NUMBER_RE.findall(address_cf)))
        if country:
            for number in numbers:
                keys.add((12, f"{country.casefold()}\x1f{number}"))
        return [(kind, value) for kind, value in keys], numbers

    def _ingest_source(self, db: sqlite3.Connection, source_id: int, path_value: str) -> None:
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(path)
        sig = f"{self.signature}|source={source_id}"
        stage = f"person3_data:{source_id}"
        completed = db.execute("SELECT signature FROM stage_state WHERE stage=?", (stage,)).fetchone()
        if completed and completed[0] == sig:
            print(f"[{self.split}] source {source_id} already indexed; cache reused", flush=True)
            return
        progress = db.execute("SELECT signature FROM stage_state WHERE stage=?", (stage + ":progress",)).fetchone()
        count = 0
        scanned = 0
        if progress and progress[0].startswith(sig + "|"):
            try:
                scanned, count = map(int, progress[0][len(sig) + 1:].split("|", 1))
            except ValueError as exc:
                raise RuntimeError(f"Invalid source checkpoint for source {source_id}") from exc
            actual = db.execute("SELECT COUNT(*) FROM records WHERE source_id=?", (source_id,)).fetchone()[0]
            if actual != count:
                raise RuntimeError(f"Checkpoint mismatch for source {source_id}: state={count}, db={actual}")
            print(f"[{self.split}] resuming source {source_id} at {count:,} rows", flush=True)
        else:
            present = db.execute("SELECT COUNT(*) FROM records WHERE source_id=?", (source_id,)).fetchone()[0]
            if present:
                raise RuntimeError(f"Source {source_id} has {present:,} rows but no matching checkpoint")

        self.phase_started = time.perf_counter()
        checkpoint_rows = scanned
        skipped = 0
        for frame in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                                 usecols=["entity_id", "business_name", "business_address", "country"],
                                 chunksize=self.chunk_rows, encoding="utf-8"):
            if skipped < checkpoint_rows:
                drop = min(len(frame), checkpoint_rows - skipped)
                skipped += drop
                if drop == len(frame):
                    continue
                frame = frame.iloc[drop:]
            original_rows = len(frame)
            scanned += original_rows
            if self.rows_per_source is not None and count + len(frame) > self.rows_per_source:
                frame = frame.iloc[:self.rows_per_source - count]
            if source_id == 1 and self.max_s1_id is not None:
                suffix = pd.to_numeric(frame.entity_id.str[3:], errors="coerce")
                frame = frame[suffix <= self.max_s1_id]
            if frame.empty:
                continue
            cleaned = self.cleaner.preprocess_dataset(frame)
            record_batch: list[tuple[Any, ...]] = []
            key_batch: list[tuple[int, str, str, int, str]] = []
            number_batch: list[tuple[str, int, str]] = []
            for eid, raw_name, raw_address, raw_country, name, address, country in zip(
                    cleaned.entity_id, cleaned.business_name, cleaned.business_address, cleaned.country,
                    cleaned.clean_name, cleaned.clean_address, cleaned.clean_country, strict=True):
                if not eid.startswith(f"S{source_id}-"):
                    raise ValueError(f"Unexpected source {source_id} ID: {eid}")
                suffix = eid[3:]
                entity_num = int(suffix) if suffix.isdigit() else None
                if source_id == 1 and entity_num is None:
                    raise ValueError(f"Source 1 IDs must have numeric suffixes: {eid}")
                side = 1 if source_id == 1 else 2
                record_batch.append((eid, source_id, name, address, country, entity_num,
                                     raw_name, raw_address, raw_country))
                keys, numbers = self._keys_for_record(name, address, country)
                key_batch.extend((kind, _hash(value), value, side, eid) for kind, value in keys)
                number_batch.extend((eid, side, number) for number in numbers)
            with db:
                db.executemany("""INSERT OR IGNORE INTO records
                    (entity_id,source_id,clean_name,clean_address,clean_country,entity_num,raw_name,raw_address,raw_country)
                    VALUES(?,?,?,?,?,?,?,?,?)""", record_batch)
                db.executemany("INSERT OR IGNORE INTO retrieval_keys VALUES(?,?,?,?,?)", key_batch)
                db.executemany("INSERT OR IGNORE INTO record_numbers VALUES(?,?,?)", number_batch)
                count += len(record_batch)
                db.execute("INSERT OR REPLACE INTO stage_state VALUES(?,?)",
                           (stage + ":progress", f"{sig}|{scanned}|{count}"))
            if count % max(self.chunk_rows * 10, self.chunk_rows) < len(record_batch):
                self._print_progress(f"index source {source_id}", count)
            del frame, cleaned, record_batch, key_batch, number_batch
            if self.rows_per_source is not None and count >= self.rows_per_source:
                break
        db.execute("INSERT OR REPLACE INTO stage_state VALUES(?,?)", (stage, sig))
        db.execute("DELETE FROM stage_state WHERE stage=?", (stage + ":progress",))
        db.commit()
        self._print_progress(f"indexed source {source_id} complete", count)
        self.phase_times[f"index_source_{source_id}_seconds"] = time.perf_counter() - self.phase_started

    def _ingest_truth(self, db: sqlite3.Connection, path_value: str) -> None:
        path = Path(path_value)
        sig = self.signature + "|truth"
        stage = "person3_truth"
        complete = db.execute("SELECT signature FROM stage_state WHERE stage=?", (stage,)).fetchone()
        if complete and complete[0] == sig:
            print(f"[{self.split}] ground truth cache reused", flush=True)
            return
        count = 0
        self.phase_started = time.perf_counter()
        for frame in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                                 usecols=["source1_entity_id", "matched_entity_ids"],
                                 chunksize=self.chunk_rows, encoding="utf-8"):
            rows = []
            source_ids = frame.source1_entity_id.drop_duplicates().tolist()
            indexed_s1: set[str] = set()
            for start in range(0, len(source_ids), 900):
                part = source_ids[start:start + 900]
                marks = ",".join("?" for _ in part)
                indexed_s1.update(row[0] for row in db.execute(
                    f"SELECT entity_id FROM records WHERE source_id=1 AND entity_id IN ({marks})", part))
            for s1, targets in zip(frame.source1_entity_id, frame.matched_entity_ids, strict=True):
                if s1 not in indexed_s1:
                    continue
                seen = set(value.strip() for value in targets.split(",") if value.strip())
                if not s1.startswith("S1-") or any(not target.startswith(("S2-", "S3-")) for target in seen):
                    raise ValueError(f"Invalid challenge truth IDs for {s1}")
                rows.extend((s1, target) for target in seen)
            with db:
                db.executemany("INSERT OR IGNORE INTO truth VALUES(?,?)", rows)
            count += len(frame)
            del rows, frame
        db.execute("INSERT OR REPLACE INTO stage_state VALUES(?,?)", (stage, sig))
        db.commit()
        print(f"[{self.split}] ground truth rows={count:,}; pairs="
              f"{db.execute('SELECT COUNT(*) FROM truth').fetchone()[0]:,}", flush=True)
        self.phase_times["truth_index_seconds"] = time.perf_counter() - self.phase_started

    def _derive_frequency_keys(self, db: sqlite3.Connection) -> None:
        stage_sig = self.signature + "|derived-keys-v2"
        row = db.execute("SELECT signature FROM stage_state WHERE stage='person3_derived_keys'").fetchone()
        if row and row[0] == stage_sig:
            print(f"[{self.split}] corpus rarity/combination keys reused", flush=True)
            return
        self.phase_started = time.perf_counter()
        db.execute("DROP TABLE IF EXISTS person3_selected_tokens")
        db.execute("CREATE TABLE person3_selected_tokens(family INTEGER NOT NULL,entity_id TEXT NOT NULL,side INTEGER NOT NULL,key_hash TEXT NOT NULL,key_value TEXT NOT NULL,document_frequency INTEGER NOT NULL,PRIMARY KEY(family,entity_id,key_hash)) WITHOUT ROWID")
        db.execute("DROP TABLE IF EXISTS person3_token_frequency")
        db.execute("""CREATE TABLE person3_token_frequency AS
            SELECT kind,key_hash,MAX(SUM(CASE WHEN side=1 THEN 1 ELSE 0 END),SUM(CASE WHEN side=2 THEN 1 ELSE 0 END)) AS df
            FROM retrieval_keys WHERE kind IN (6,8) GROUP BY kind,key_hash""")
        db.execute("CREATE UNIQUE INDEX person3_token_frequency_key ON person3_token_frequency(kind,key_hash)")
        for source_kind, selected_family, keep in ((6, 7, self.name_tokens_per_record),
                                                    (8, 9, self.address_tokens_per_record)):
            sql = """WITH ranked AS (
                SELECT k.entity_id,k.side,k.key_hash,k.key_value,f.df,
                  ROW_NUMBER() OVER(PARTITION BY k.entity_id ORDER BY f.df,k.key_hash) AS rank
                FROM retrieval_keys k JOIN person3_token_frequency f
                  ON f.kind=k.kind AND f.key_hash=k.key_hash WHERE k.kind=?
              ) INSERT INTO person3_selected_tokens
                SELECT ?,entity_id,side,key_hash,key_value,df FROM ranked WHERE rank<=?"""
            db.execute(sql, (source_kind, selected_family, keep))
        db.execute("""INSERT OR IGNORE INTO retrieval_keys(kind,key_hash,key_value,side,entity_id)
            SELECT 7,key_hash,key_value,side,entity_id FROM person3_selected_tokens WHERE family=7""")
        db.execute("""INSERT OR IGNORE INTO retrieval_keys(kind,key_hash,key_value,side,entity_id)
            SELECT 9,key_hash,key_value,side,entity_id FROM person3_selected_tokens WHERE family=9""")
        # Every pair of at most six rare name tokens gives an order-invariant,
        # high-information signature; this is bounded at 15 keys per entity.
        db.execute("""INSERT OR IGNORE INTO retrieval_keys(kind,key_hash,key_value,side,entity_id)
            SELECT 14,a.key_hash||b.key_hash,a.key_hash||b.key_hash,a.side,a.entity_id
            FROM person3_selected_tokens a JOIN person3_selected_tokens b
              ON a.family=7 AND b.family=7 AND a.entity_id=b.entity_id AND a.side=b.side
             AND a.key_hash<b.key_hash""")
        db.execute("""INSERT OR IGNORE INTO retrieval_keys(kind,key_hash,key_value,side,entity_id)
            SELECT 15,n.key_hash||a.key_hash,n.key_hash||a.key_hash,n.side,n.entity_id
            FROM person3_selected_tokens n JOIN person3_selected_tokens a
              ON n.family=7 AND a.family=9 AND n.entity_id=a.entity_id AND n.side=a.side""")
        # Join selected informative name tokens to each stored street number.
        db.execute("""INSERT OR IGNORE INTO retrieval_keys(kind,key_hash,key_value,side,entity_id)
            SELECT 13,n.key_hash||'|'||r.number,n.key_hash||'|'||r.number,n.side,n.entity_id
            FROM person3_selected_tokens n JOIN record_numbers r
              ON n.family=7 AND n.entity_id=r.entity_id AND n.side=r.side""")
        # Country-qualified variants refine broad evidence without making
        # country a global hard filter. Base and cross-country families remain.
        db.execute("""INSERT OR IGNORE INTO retrieval_keys(kind,key_hash,key_value,side,entity_id)
            SELECT 16,LOWER(TRIM(r.clean_country))||char(31)||n.key_hash,
                   LOWER(TRIM(r.clean_country))||char(31)||n.key_value,n.side,n.entity_id
            FROM person3_selected_tokens n JOIN records r ON r.entity_id=n.entity_id
            WHERE n.family=7 AND TRIM(r.clean_country)<>''""")
        db.execute("""INSERT OR IGNORE INTO retrieval_keys(kind,key_hash,key_value,side,entity_id)
            SELECT 17,LOWER(TRIM(r.clean_country))||char(31)||k.key_hash,
                   LOWER(TRIM(r.clean_country))||char(31)||k.key_value,k.side,k.entity_id
            FROM retrieval_keys k JOIN records r ON r.entity_id=k.entity_id
            WHERE k.kind=8 AND TRIM(r.clean_country)<>''""")
        db.execute("""INSERT OR IGNORE INTO retrieval_keys(kind,key_hash,key_value,side,entity_id)
            SELECT 18,k.key_hash||'|'||n.number,k.key_value||char(31)||n.number,k.side,k.entity_id
            FROM person3_selected_tokens k JOIN record_numbers n
              ON k.family=9 AND k.entity_id=n.entity_id AND k.side=n.side""")
        db.execute("INSERT OR REPLACE INTO stage_state VALUES('person3_derived_keys',?)", (stage_sig,))
        db.commit()
        total = db.execute("SELECT COUNT(*) FROM retrieval_keys").fetchone()[0]
        self._print_progress("derived rare-token and combination keys", total)
        self.phase_times["derive_keys_seconds"] = time.perf_counter() - self.phase_started

    def _build_block_stats(self, db: sqlite3.Connection) -> None:
        stage_sig = f"{self.signature}|max_pairs={self.max_block_pairs}|stats-v2"
        row = db.execute("SELECT signature FROM stage_state WHERE stage='person3_block_stats'").fetchone()
        if row and row[0] == stage_sig:
            return
        self.phase_started = time.perf_counter()
        db.execute("DELETE FROM person3_block_stats")
        db.execute("""INSERT INTO person3_block_stats
            SELECT kind,key_hash,
              SUM(CASE WHEN side=1 THEN 1 ELSE 0 END),
              SUM(CASE WHEN side=2 THEN 1 ELSE 0 END),
              SUM(CASE WHEN side=1 THEN 1 ELSE 0 END)*SUM(CASE WHEN side=2 THEN 1 ELSE 0 END),
              CASE WHEN SUM(CASE WHEN side=1 THEN 1 ELSE 0 END)*SUM(CASE WHEN side=2 THEN 1 ELSE 0 END)<=? THEN 1 ELSE 0 END
            FROM retrieval_keys GROUP BY kind,key_hash
            HAVING SUM(CASE WHEN side=1 THEN 1 ELSE 0 END)>0
               AND SUM(CASE WHEN side=2 THEN 1 ELSE 0 END)>0""", (self.max_block_pairs,))
        db.execute("INSERT OR REPLACE INTO stage_state VALUES('person3_block_stats',?)", (stage_sig,))
        db.commit()
        row = db.execute("SELECT COUNT(*),SUM(included),SUM(CASE WHEN included=0 THEN pair_count ELSE 0 END) FROM person3_block_stats").fetchone()
        self._print_progress("block sizes indexed", int(row[0] or 0),
                             f"safe={row[1] or 0:,}; theoretical_omitted_pairs={row[2] or 0:,}")
        self.phase_times["block_stats_seconds"] = time.perf_counter() - self.phase_started

    @staticmethod
    def _candidate_bit(kind: int, block_flag: int) -> tuple[int, int]:
        return block_flag, RETRIEVAL_BITS[kind]

    @staticmethod
    def _s1_number_ranges(db: sqlite3.Connection, *, after: int = -1,
                         batch_entities: int = 20_000) -> Iterable[tuple[int, int, int]]:
        """Yield bounded min/max ranges over actual S1 IDs, never numeric gaps."""
        cursor = after
        while True:
            batch = [row[0] for row in db.execute(
                "SELECT entity_num FROM records WHERE source_id=1 AND entity_num>? "
                "ORDER BY entity_num LIMIT ?", (cursor, batch_entities))]
            if not batch:
                return
            yield int(batch[0]), int(batch[-1]), len(batch)
            cursor = int(batch[-1])

    def _generate_family(self, db: sqlite3.Connection, kind: int, s1_min: int, s1_max: int,
                         batch_entities: int = 20_000) -> dict[str, int | float]:
        name, block_flag = RETRIEVAL_FAMILIES[kind]
        state_name = f"person3_candidates:{kind}"
        state_sig = f"{self.signature}|{self.max_block_pairs}|{kind}|{s1_min}|{s1_max}"
        state = db.execute("SELECT signature FROM stage_state WHERE stage=?", (state_name,)).fetchone()
        last = s1_min - 1
        if state and state[0].startswith(state_sig + "|"):
            last = int(state[0].rsplit("|", 1)[1])
            if last >= s1_max:
                count = db.execute("SELECT COUNT(*) FROM candidates WHERE (retrieval_flags & ?) != 0",
                                   (RETRIEVAL_BITS[kind],)).fetchone()[0]
                return {"candidate_pairs": int(count), "resumed_at": int(last)}
        before = db.execute("SELECT COUNT(*) FROM candidates WHERE (retrieval_flags & ?) != 0",
                            (RETRIEVAL_BITS[kind],)).fetchone()[0]
        self.phase_started = time.perf_counter()
        cursor = max(s1_min, last + 1)
        processed_entities = 0
        batch_no = 0
        query = """INSERT INTO candidates(source1_entity_id,target_entity_id,block_flags,retrieval_flags)
            SELECT DISTINCT r.entity_id,t.entity_id,?,?
            FROM retrieval_keys r JOIN person3_block_stats b
              ON b.kind=r.kind AND b.key_hash=r.key_hash AND b.included=1
            JOIN retrieval_keys t ON t.kind=r.kind AND t.key_hash=r.key_hash AND t.side=2
            JOIN records s ON s.entity_id=r.entity_id
            JOIN records z ON z.entity_id=t.entity_id
            WHERE r.kind=? AND r.side=1 AND s.entity_num BETWEEN ? AND ?
            ON CONFLICT(source1_entity_id,target_entity_id) DO UPDATE SET
              block_flags=candidates.block_flags | excluded.block_flags,
              retrieval_flags=candidates.retrieval_flags | excluded.retrieval_flags"""
        for lower, upper, entity_count in self._s1_number_ranges(
                db, after=cursor - 1, batch_entities=batch_entities):
            if upper < cursor:
                continue
            with db:
                db.execute(query, (block_flag, RETRIEVAL_BITS[kind], kind, lower, upper))
                db.execute("INSERT OR REPLACE INTO stage_state VALUES(?,?)",
                           (state_name, f"{state_sig}|{upper}"))
            cursor = upper + 1
            processed_entities += entity_count
            batch_no += 1
            if batch_no % 10 == 0:
                n = db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
                self._print_progress(f"retrieve {name}", processed_entities,
                                     f"candidates={n:,}; id_range_end={upper:,}")
        after = db.execute("SELECT COUNT(*) FROM candidates WHERE (retrieval_flags & ?) != 0",
                           (RETRIEVAL_BITS[kind],)).fetchone()[0]
        db.commit()
        elapsed = time.perf_counter() - self.phase_started
        return {"candidate_pairs": int(after), "new_pair_attributions": int(after - before),
                "elapsed_seconds": elapsed,
                "candidate_pairs_per_second": int(after / max(elapsed, 1e-6))}

    def _generate_fuzzy_fallback(self, db: sqlite3.Connection, s1_min: int, s1_max: int,
                                 batch_entities: int = 20_000, batch_size: int = 5_000) -> dict[str, int | float]:
        """Run fuzzy scoring only in safe four-character token-prefix blocks.

        It is an adaptive fallback for S1s with fewer than the configured number
        of exact/token/address candidates. The block cross product is bounded by
        the same explicit pair budget as all other retrieval families.
        """
        kind = 10
        name = "bounded_fuzzy_fallback"
        flag = RETRIEVAL_BITS[kind]
        block_flag = RETRIEVAL_FAMILIES[kind][1]
        state_name = "person3_candidates:fuzzy"
        state_sig = (f"{self.signature}|{self.max_block_pairs}|{self.fuzzy_threshold}|"
                     f"{self.fuzzy_trigger_candidate_count}|{s1_min}|{s1_max}")
        state = db.execute("SELECT signature FROM stage_state WHERE stage=?", (state_name,)).fetchone()
        cursor = s1_min
        if state and state[0].startswith(state_sig + "|"):
            cursor = int(state[0].rsplit("|", 1)[1]) + 1
            if cursor > s1_max:
                return {"evaluated_pairs": 0, "retrieved_pairs": int(db.execute(
                    "SELECT COUNT(*) FROM candidates WHERE (retrieval_flags & ?) != 0", (flag,)).fetchone()[0]),
                    "elapsed_seconds": 0.0}
        self.phase_started = time.perf_counter()
        evaluated = retrieved = 0
        db.execute("DROP TABLE IF EXISTS temp.person3_fuzzy_s1")
        db.execute("""CREATE TEMP TABLE person3_fuzzy_s1 AS
            SELECT r.entity_id FROM records r LEFT JOIN
              (SELECT source1_entity_id,COUNT(*) n FROM candidates GROUP BY source1_entity_id) c
              ON c.source1_entity_id=r.entity_id
            WHERE r.source_id=1 AND COALESCE(c.n,0)<?""",
            (self.fuzzy_trigger_candidate_count,))
        db.execute("CREATE UNIQUE INDEX temp.person3_fuzzy_s1_id ON person3_fuzzy_s1(entity_id)")
        insert_sql = """INSERT INTO candidates(source1_entity_id,target_entity_id,block_flags,retrieval_flags)
            VALUES(?,?,?,?) ON CONFLICT(source1_entity_id,target_entity_id) DO UPDATE SET
              block_flags=candidates.block_flags | excluded.block_flags,
              retrieval_flags=candidates.retrieval_flags | excluded.retrieval_flags"""
        batches = 0
        for lower, upper, _ in self._s1_number_ranges(db, after=cursor - 1,
                                                      batch_entities=batch_entities):
            if upper < cursor:
                continue
            q = db.execute("""SELECT DISTINCT s.entity_id,t.entity_id,s.clean_name,z.clean_name
                FROM retrieval_keys r JOIN person3_block_stats b
                  ON b.kind=r.kind AND b.key_hash=r.key_hash AND b.included=1
                JOIN retrieval_keys t ON t.kind=r.kind AND t.key_hash=r.key_hash AND t.side=2
                JOIN records s ON s.entity_id=r.entity_id
                JOIN records z ON z.entity_id=t.entity_id
                JOIN person3_fuzzy_s1 eligible ON eligible.entity_id=s.entity_id
                WHERE r.kind=10 AND r.side=1 AND s.entity_num BETWEEN ? AND ?
                """, (lower, upper))
            batch = []
            while True:
                rows = q.fetchmany(batch_size)
                if not rows:
                    break
                for s1, target, s1_name, target_name in rows:
                    evaluated += 1
                    score = fuzz.WRatio(s1_name, target_name) if s1_name and target_name else 0
                    if score >= self.fuzzy_threshold:
                        batch.append((s1, target, block_flag, flag))
                if batch:
                    with db:
                        db.executemany(insert_sql, batch)
                    retrieved += len(batch)
                    batch.clear()
            with db:
                db.execute("INSERT OR REPLACE INTO stage_state VALUES(?,?)",
                           (state_name, f"{state_sig}|{upper}"))
            cursor = upper + 1
            batches += 1
            if batches % 10 == 0:
                self._print_progress(name, evaluated,
                                     f"fuzzy_hits={retrieved:,}; id_range_end={upper:,}")
        elapsed = time.perf_counter() - self.phase_started
        self.phase_times["bounded_fuzzy_fallback_seconds"] = elapsed
        return {"evaluated_pairs": evaluated, "new_retrieved_pairs": retrieved,
                "threshold": self.fuzzy_threshold, "trigger_candidates_below": self.fuzzy_trigger_candidate_count,
                "elapsed_seconds": elapsed,
                "evaluated_pairs_per_second": int(evaluated / max(elapsed, 1e-6))}

    def _build_candidates(self, db: sqlite3.Connection) -> dict[str, dict[str, int]]:
        self._derive_frequency_keys(db)
        self._build_block_stats(db)
        bounds = db.execute("SELECT MIN(entity_num),MAX(entity_num) FROM records WHERE source_id=1").fetchone()
        if bounds[0] is None:
            raise ValueError("Candidate retrieval requires Source 1 records")
        summary = {}
        for kind in RETRIEVAL_FAMILIES:
            if kind == 10 or kind in self.disabled_families:
                continue  # fuzzy is an adaptive, scored fallback below
            summary[RETRIEVAL_FAMILIES[kind][0]] = self._generate_family(db, kind, int(bounds[0]), int(bounds[1]))
        summary["bounded_fuzzy_fallback"] = self._generate_fuzzy_fallback(
            db, int(bounds[0]), int(bounds[1]))
        db.execute("CREATE INDEX IF NOT EXISTS person3_candidates_target ON candidates(target_entity_id)")
        db.commit()
        return summary

    def _audit(self, db: sqlite3.Connection) -> dict[str, Any]:
        if self.split != "train":
            return {"status": "not_applicable_test_has_no_ground_truth"}
        self.phase_started = time.perf_counter()
        truth_count = int(db.execute("SELECT COUNT(*) FROM truth").fetchone()[0])
        available_truth_count = int(db.execute("""SELECT COUNT(*) FROM truth t JOIN records r
            ON r.entity_id=t.target_entity_id AND r.source_id IN (2,3)""").fetchone()[0])
        hit_count = int(db.execute("SELECT COUNT(*) FROM truth t JOIN candidates c USING(source1_entity_id,target_entity_id)").fetchone()[0])
        positives = db.execute("""SELECT COUNT(*),SUM(CASE WHEN COALESCE(c.hit,0)=g.n THEN 1 ELSE 0 END),
              SUM(CASE WHEN COALESCE(c.hit,0)=0 THEN 1 ELSE 0 END)
            FROM (SELECT source1_entity_id,COUNT(*) n FROM truth GROUP BY source1_entity_id) g
            LEFT JOIN (SELECT t.source1_entity_id,COUNT(*) hit FROM truth t JOIN candidates c USING(source1_entity_id,target_entity_id) GROUP BY t.source1_entity_id) c USING(source1_entity_id)""").fetchone()
        entity_coverage = db.execute("""SELECT COUNT(DISTINCT t.source1_entity_id)
            FROM truth t JOIN candidates c USING(source1_entity_id,target_entity_id)""").fetchone()[0]
        target_available_entities = db.execute("""SELECT COUNT(DISTINCT t.source1_entity_id)
            FROM truth t JOIN records r ON r.entity_id=t.target_entity_id AND r.source_id IN (2,3)""").fetchone()[0]
        target_available_entity_hits = db.execute("""SELECT COUNT(DISTINCT t.source1_entity_id)
            FROM truth t JOIN records r ON r.entity_id=t.target_entity_id AND r.source_id IN (2,3)
            JOIN candidates c USING(source1_entity_id,target_entity_id)""").fetchone()[0]
        by_source = {}
        for prefix in ("S2-", "S3-"):
            total = db.execute("SELECT COUNT(*) FROM truth WHERE target_entity_id LIKE ?", (prefix + "%",)).fetchone()[0]
            got = db.execute("SELECT COUNT(*) FROM truth t JOIN candidates c USING(source1_entity_id,target_entity_id) WHERE t.target_entity_id LIKE ?", (prefix + "%",)).fetchone()[0]
            by_source[prefix[:2]] = {"true_pairs": int(total), "recovered_pairs": int(got), "recall": got / max(1, total)}
        by_country = {}
        for country, total, got in db.execute("""SELECT r.clean_country,COUNT(*),SUM(CASE WHEN c.source1_entity_id IS NULL THEN 0 ELSE 1 END)
            FROM truth t JOIN records r ON r.entity_id=t.source1_entity_id
            LEFT JOIN candidates c USING(source1_entity_id,target_entity_id)
            GROUP BY r.clean_country ORDER BY r.clean_country"""):
            by_country[country] = {"true_pairs": int(total), "recovered_pairs": int(got), "recall": got / max(1,total)}
        by_address = {}
        for label, condition in (("s1_address_present", "r.clean_address<>''"), ("s1_address_missing", "r.clean_address=''")):
            total, got = db.execute(f"""SELECT COUNT(*),SUM(CASE WHEN c.source1_entity_id IS NULL THEN 0 ELSE 1 END)
                FROM truth t JOIN records r ON r.entity_id=t.source1_entity_id
                LEFT JOIN candidates c USING(source1_entity_id,target_entity_id) WHERE {condition}""").fetchone()
            by_address[label] = {"true_pairs": int(total or 0), "recovered_pairs": int(got or 0), "recall": (got or 0) / max(1,total or 0)}
        for label, condition in (("target_address_present", "z.clean_address<>''"),
                                 ("target_address_missing", "z.clean_address=''")):
            total, got = db.execute(f"""SELECT COUNT(*),SUM(CASE WHEN c.source1_entity_id IS NULL THEN 0 ELSE 1 END)
                FROM truth t JOIN records z ON z.entity_id=t.target_entity_id
                LEFT JOIN candidates c USING(source1_entity_id,target_entity_id) WHERE {condition}""").fetchone()
            by_address[label] = {"true_pairs": int(total or 0), "recovered_pairs": int(got or 0), "recall": (got or 0) / max(1,total or 0)}
        by_strategy = {}
        for kind, (name, _) in RETRIEVAL_FAMILIES.items():
            bit = RETRIEVAL_BITS[kind]
            got = db.execute("SELECT COUNT(*) FROM truth t JOIN candidates c USING(source1_entity_id,target_entity_id) WHERE (c.retrieval_flags & ?) != 0", (bit,)).fetchone()[0]
            by_strategy[name] = {"recovered_true_pairs": int(got), "recall_contribution": got / max(1, truth_count)}
        no_match, no_match_zero, no_match_nonzero = db.execute("""SELECT COUNT(*),
            SUM(CASE WHEN COALESCE(c.n,0)=0 THEN 1 ELSE 0 END),
            SUM(CASE WHEN COALESCE(c.n,0)>0 THEN 1 ELSE 0 END)
            FROM records r LEFT JOIN (SELECT source1_entity_id,COUNT(*) n FROM truth GROUP BY source1_entity_id) t ON t.source1_entity_id=r.entity_id
            LEFT JOIN (SELECT source1_entity_id,COUNT(*) n FROM candidates GROUP BY source1_entity_id) c ON c.source1_entity_id=r.entity_id
            WHERE r.source_id=1 AND COALESCE(t.n,0)=0""").fetchone()
        db.execute("DROP TABLE IF EXISTS temp.person3_candidate_count_distribution")
        db.execute("""CREATE TEMP TABLE person3_candidate_count_distribution AS
            SELECT COALESCE(c.n,0) n FROM records r LEFT JOIN
              (SELECT source1_entity_id,COUNT(*) n FROM candidates GROUP BY source1_entity_id) c
              ON c.source1_entity_id=r.entity_id WHERE r.source_id=1""")
        db.execute("CREATE INDEX temp.person3_candidate_count_distribution_n ON person3_candidate_count_distribution(n)")
        bucket = {"0": 0, "1": 0, "2-5": 0, "6-10": 0, "11-20": 0, "21-50": 0,
                  "51-100": 0, "101-500": 0, "501-1000": 0, "1001+": 0}
        for (n,) in db.execute("SELECT n FROM person3_candidate_count_distribution"):
            key = ("0" if n == 0 else "1" if n == 1 else "2-5" if n <= 5 else "6-10" if n <= 10
                   else "11-20" if n <= 20 else "21-50" if n <= 50 else "51-100" if n <= 100
                   else "101-500" if n <= 500 else "501-1000" if n <= 1000 else "1001+")
            bucket[key] += 1
        def quantile(p: float) -> int:
            n = sum(bucket.values())
            if not n: return 0
            row = db.execute("SELECT n FROM person3_candidate_count_distribution ORDER BY n LIMIT 1 OFFSET ?",
                             (min(n-1, int((n-1)*p)),)).fetchone()
            return int(row[0]) if row else 0
        total_candidates = int(db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
        missed = truth_count - hit_count
        db.execute("DELETE FROM person3_misses")
        db.execute("""INSERT INTO person3_misses
            SELECT t.source1_entity_id,t.target_entity_id,CASE WHEN t.target_entity_id LIKE 'S2-%' THEN 2 ELSE 3 END,
              s.raw_name,s.raw_address,s.raw_country,s.clean_name,s.clean_address,s.clean_country,
              COALESCE(z.raw_name,''),COALESCE(z.raw_address,''),COALESCE(z.raw_country,''),
              COALESCE(z.clean_name,''),COALESCE(z.clean_address,''),COALESCE(z.clean_country,''),
              COALESCE((SELECT GROUP_CONCAT(DISTINCT r.kind) FROM retrieval_keys r JOIN retrieval_keys q
                ON q.kind=r.kind AND q.key_hash=r.key_hash WHERE r.entity_id=t.source1_entity_id AND q.entity_id=t.target_entity_id),''),
              CASE WHEN z.entity_id IS NULL THEN 'true target absent from the indexed target corpus'
                WHEN EXISTS(SELECT 1 FROM retrieval_keys r JOIN retrieval_keys q
                ON q.kind=r.kind AND q.key_hash=r.key_hash WHERE r.entity_id=t.source1_entity_id AND q.entity_id=t.target_entity_id)
                THEN 'shared key exists but block exceeded configured pair budget or retrieval stage did not complete'
                ELSE 'no shared generated retrieval key' END
            FROM truth t JOIN records s ON s.entity_id=t.source1_entity_id
            LEFT JOIN records z ON z.entity_id=t.target_entity_id AND z.source_id IN (2,3)
            LEFT JOIN candidates c USING(source1_entity_id,target_entity_id)
            WHERE c.source1_entity_id IS NULL""")
        db.commit()
        miss_path = self.root / "reports" / "candidate_recall_misses.tsv"
        miss_path.parent.mkdir(parents=True, exist_ok=True)
        with miss_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow([d[1] for d in db.execute("PRAGMA table_info(person3_misses)")])
            writer.writerows(db.execute("SELECT * FROM person3_misses ORDER BY source1_entity_id,target_entity_id"))
        report = {
            "status": "measured", "split": "train", "true_match_pairs": truth_count,
            "recovered_true_match_pairs": hit_count, "missed_true_match_pairs": missed,
            "pair_candidate_recall": hit_count / max(1, truth_count),
            "true_pairs_with_target_present": available_truth_count,
            "recovered_pair_recall_when_target_present": hit_count / max(1, available_truth_count),
            "positive_source1_entities": int(positives[0] or 0),
            "all_match_entity_coverage": (positives[1] or 0) / max(1, positives[0] or 0),
            "at_least_one_match_entity_coverage": int(entity_coverage) / max(1, int(positives[0] or 0)),
            "true_positive_entities_with_target_present": int(target_available_entities),
            "at_least_one_match_entity_coverage_when_target_present":
                int(target_available_entity_hits) / max(1, int(target_available_entities)),
            "positive_entities_with_no_true_candidate": int(positives[2] or 0),
            "no_match_source1_entities": int(no_match or 0),
            "no_match_zero_candidate_entities": int(no_match_zero or 0),
            "no_match_nonzero_candidate_entities": int(no_match_nonzero or 0),
            "no_match_candidate_rate": (no_match_nonzero or 0) / max(1, no_match or 0),
            "source_recall": by_source, "country_recall": by_country,
            "address_presence_recall": by_address, "strategy_true_pair_recall": by_strategy,
            "candidate_count": total_candidates,
            "candidate_count_per_source1": {"mean": total_candidates / max(1, sum(bucket.values())),
                "median": quantile(.5), "p90": quantile(.9), "p95": quantile(.95),
                "p99": quantile(.99), "max": quantile(1.0),
                "buckets": bucket},
            "misses_tsv": str(miss_path),
        }
        report_path = self.root / "reports" / "person3_candidate_metrics.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def build(self, split: str, source_paths: dict[str, str], *, output_db: str | Path,
              truth_path: str | None = None) -> dict[str, Any]:
        if split not in {"train", "test"}:
            raise ValueError("split must be train or test")
        if split == "train" and not truth_path:
            raise ValueError("Training retrieval requires the official ground-truth TSV")
        self.split = split
        self.input_paths = dict(source_paths)
        if split == "train" and truth_path:
            self.input_paths["ground_truth"] = truth_path
        options = {"chunk_rows": self.chunk_rows, "max_block_pairs": self.max_block_pairs,
                   "name_tokens_per_record": self.name_tokens_per_record,
                   "address_tokens_per_record": self.address_tokens_per_record,
                   "rows_per_source": self.rows_per_source, "max_s1_id": self.max_s1_id,
                   "fuzzy_threshold": self.fuzzy_threshold,
                   "fuzzy_trigger_candidate_count": self.fuzzy_trigger_candidate_count,
                   "disabled_families": sorted(self.disabled_families)}
        self.signature = self._file_signature(self.input_paths, options)
        self.root.mkdir(parents=True, exist_ok=True)
        initial_usage = shutil.disk_usage(self.root)
        self.disk_free_start = initial_usage.free
        self.disk_free_minimum = initial_usage.free
        self.rss_observed_peak_mib = self.process.memory_info().rss / (1024 * 1024)
        self.db_path = Path(output_db).resolve()
        self.started = time.perf_counter()
        db = self._connect(self.db_path)
        try:
            self._check_signature(db)
            for source_id in (1, 2, 3):
                key = f"source{source_id}"
                self._ingest_source(db, source_id, source_paths[key])
            if split == "train":
                self._ingest_truth(db, truth_path)
            db.execute("CREATE INDEX IF NOT EXISTS person3_records_source_num ON records(source_id,entity_num)")
            db.execute("CREATE INDEX IF NOT EXISTS person3_key_lookup ON retrieval_keys(kind,key_hash,side,entity_id)")
            db.commit()
            candidate_started = time.perf_counter()
            rules = self._build_candidates(db)
            candidate_seconds = time.perf_counter() - candidate_started
            block_total, block_included, omitted_pairs = db.execute("""SELECT COUNT(*),SUM(included),
                SUM(CASE WHEN included=0 THEN pair_count ELSE 0 END) FROM person3_block_stats""").fetchone()
            if split == "train":
                index_started = time.perf_counter()
                db.execute("CREATE INDEX IF NOT EXISTS person3_key_by_entity ON retrieval_keys(entity_id,kind,key_hash)")
                db.commit()
                self.phase_times["miss_audit_entity_index_seconds"] = time.perf_counter() - index_started
                recall = self._audit(db)
            else:
                recall = {"status": "not_applicable_test_has_no_ground_truth"}
            counts = {f"source{source_id}_records": int(db.execute(
                "SELECT COUNT(*) FROM records WHERE source_id=?", (source_id,)).fetchone()[0])
                for source_id in (1, 2, 3)}
            candidate_count = int(db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
            output = {"status": "measured", "split": split, "database": str(self.db_path),
                "input_paths": self.input_paths, "input_records": counts,
                "candidate_pairs": candidate_count, "retrieval_rules": rules,
                "max_block_pairs": self.max_block_pairs, "candidate_generation_seconds": candidate_seconds,
                "phase_timings_seconds": self.phase_times,
                "total_elapsed_seconds": time.perf_counter() - self.started,
                "rss_mib": self.process.memory_info().rss / (1024 * 1024),
                "rss_observed_peak_mib": self.rss_observed_peak_mib,
                "disk_free_start_gib": self.disk_free_start / (1024 ** 3),
                "disk_free_min_observed_gib": self.disk_free_minimum / (1024 ** 3),
                "free_disk_gib": shutil.disk_usage(self.root).free / (1024 ** 3),
                "block_quality": {"blocks_total": int(block_total or 0),
                    "blocks_within_pair_budget": int(block_included or 0),
                    "blocks_over_pair_budget": int((block_total or 0) - (block_included or 0)),
                    "theoretical_pairs_omitted_by_budget": int(omitted_pairs or 0)},
                "resume_signature": self.signature, "recall_audit": recall}
            stage = "person3_complete"
            db.execute("INSERT OR REPLACE INTO stage_state VALUES(?,?)", (stage, self.signature))
            db.commit()
            report = self.root / "reports" / f"person3_{split}_build.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(output, indent=2), encoding="utf-8")
            print(f"[{split}] Person 3 complete: records={sum(counts.values()):,}; candidates={candidate_count:,}; "
                  f"elapsed={output['total_elapsed_seconds']:.1f}s; RSS={output['rss_mib']:.0f} MiB; "
                  f"D_free={output['free_disk_gib']:.1f} GiB", flush=True)
            return output
        finally:
            db.close()


def generate_candidate_tsv(database: str | Path, output_path: str | Path) -> dict[str, int]:
    """Stream Stage 5-compatible candidate output, retaining zero-candidate S1s."""
    db = sqlite3.connect(f"file:{Path(database).resolve().as_posix()}?mode=ro", uri=True)
    count = 0
    pairs = 0
    zero_candidates = 0
    with Path(output_path).open("w", encoding="utf-8", newline="") as stream:
        stream.write("source1_entity_id\tcandidate_entity_ids\n")
        cursor = db.execute("SELECT entity_id FROM records WHERE source_id=1 ORDER BY entity_id")
        pair_cursor = db.execute("SELECT source1_entity_id,target_entity_id FROM candidates ORDER BY source1_entity_id,target_entity_id")
        current = next(pair_cursor, None)
        for (s1,) in cursor:
            stream.write(s1 + "\t")
            first = True
            while current and current[0] < s1:
                current = next(pair_cursor, None)
            while current and current[0] == s1:
                if not first:
                    stream.write(",")
                stream.write(current[1])
                first = False
                current = next(pair_cursor, None)
                pairs += 1
            if first:
                zero_candidates += 1
            stream.write("\n")
            count += 1
    invalid = db.execute("""SELECT COUNT(*) FROM candidates c JOIN records r
        ON r.entity_id=c.target_entity_id WHERE r.source_id NOT IN (2,3)
          OR (r.source_id=2 AND c.target_entity_id NOT LIKE 'S2-%')
          OR (r.source_id=3 AND c.target_entity_id NOT LIKE 'S3-%')""").fetchone()[0]
    duplicates = db.execute("SELECT COUNT(*)-COUNT(DISTINCT source1_entity_id||char(31)||target_entity_id) FROM candidates").fetchone()[0]
    expected_refs = db.execute("SELECT COUNT(*) FROM records WHERE source_id=1").fetchone()[0]
    db.close()
    if count != expected_refs or invalid or duplicates:
        raise ValueError(f"Candidate TSV validation failed: rows={count}/{expected_refs}, invalid={invalid}, duplicates={duplicates}")
    return {"source1_rows": count, "candidate_pairs": pairs,
            "zero_candidate_source1": zero_candidates, "invalid_target_ids": int(invalid),
            "duplicate_pairs": int(duplicates)}
