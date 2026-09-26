# Person 3 retrieval audit (before implementation)

Audit date: 2026-09-26. Scope: upstream retrieval and recall only. Stage 3–5 behavior was inspected for compatibility; it is not being redesigned.

## Existing data and checkpoint state

- Raw official files are present under `D:/AmazonMLChallenge2026/data/{train,test}` and match the challenge schemas. File headers are `entity_id, business_name, business_address, country`; ground truth is `source1_entity_id, matched_entity_ids`.
- Observed full row counts from the repository's current data report/config are train S1/S2/S3 = 2,206,821 / 5,034,616 / 5,285,603 and test S1/S2/S3 = 1,732,544 / 4,887,273 / 5,082,316. The full corpus exceeds 12M source rows.
- `artifacts/cache/train.sqlite` in the repository is an actual Git LFS-filtered 429,584,384-byte SQLite database, not an LFS pointer. Read-only inspection found 100,000 records, 1,216,040 candidates, 61,255 truth pairs, and 1,216,040 features. Its `feature_schema` marker is Person 2 `feature-v5`. This file must remain unchanged.
- The separate full-run checkpoint at `D:/AmazonMLChallenge2026/artifacts/cache/train.sqlite` is 4,416,061,440 bytes and contains 4,106,821 records, zero candidates/features/truth, completed S1 ingestion, and Source 2 progress at 1,850,000. It is also preserved.
- The D: benchmark DB has 3,277,137 candidates but prior run notes identify synthetic-label contamination; it is excluded from recall claims. The engine-comparison 100k DB has 1,216,040 candidates but no truth/features. No test candidate cache exists.

## Person 1/current upstream blocking

Preprocessing is performed by `TextCleaner` from `src/preprocessing/text_cleaner.py`: lower/casefold-like lowercase, punctuation-to-space, configured deterministic suffix/address term expansions, whitespace collapse, and open-set country lowercase. It does not currently generate Unicode transliteration, compact-name, or separate token-signature fields. It retains normalized text in SQLite; raw source text is not retained by the full disk runner.

The active full-run SQL retrieval is `DiskPipeline._blocking_keys` plus `generate_candidates` in `src/pipeline/disk_runner.py`, rather than the in-memory `AttributeBlocker` helper. It creates six country-scoped families: exact normalized name, name tokens, name prefixes (3/5/7 characters), address-number plus name token, exact normalized address, and address tokens. It calculates cross-source document frequencies in SQLite, discards any group whose S1 or target side exceeds `blocking.max_group_size` (currently 100), chooses at most two safe name-token blocks and one safe address-token block per S1, and unions matches into a `(source1_entity_id,target_entity_id)` primary key with OR-ed provenance bits. Exact name, prefixes, number, and exact address use all matching safe keys. Candidate generation is deterministic and has no per-entity top-K cap; oversized blocks are silently omitted from the candidate set after aggregate block counts are computed.

The standalone `AttributeBlocker` in `src/blocking/attribute_blocking.py` is an in-memory helper for small frames. It uses exact name, up to three informative name tokens, and address number plus one name token, with configurable block/pair limits and optional candidate cap. It is not the full-scale DB path.

Current recall reporting in `src/pipeline/disk_runner.py` and `src/util/candidate_recall.py` measures pair recall, all-match entity coverage, source and country recall, and basic candidate distribution. The disk report does not yet produce detailed missed-pair raw/normalized attribute diagnostics, address-presence or candidate-bucket strata, attempted-rule explanations, or no-match candidate-rate statistics.

## SQLite and Stage 3–5 interface

The full-run schema has `records(entity_id,source_id,clean_name,clean_address,clean_country)`, `block_keys(kind,block_key,side,entity_id)`, `candidates(source1_entity_id,target_entity_id,block_flags)`, `truth(source1_entity_id,target_entity_id)`, and numeric `features` keyed by S1/target pair. Candidate provenance is a bitmask; current known bits are `exact_name=1`, `name_token=2`, `name_prefix=4`, `address_number=8`, `exact_address=16`, and `address_token=32`.

Person 2's current CLI is `train|predict --from-candidates --candidate-db <SQLite> --artifact-root <path> --model-path <path>`. The candidate-only runner requires existing `records`, `candidates`, and `stage_state`, plus `truth` for training. Stage 3 reads `candidates` and joins IDs to normalized `records`; it currently ignores other candidate columns and consumes `block_flags` as a feature. Stage 5 writes `candidate_pairs.tsv` as exactly `source1_entity_id<TAB>candidate_entity_ids`, one sorted row per `records.source_id=1` record, preserving empty rows. The DB contract can therefore be extended with retrieval audit tables/columns as long as these columns, IDs, and `block_flags` remain stable.

## Retrieval gap / implementation target

The largest recall risk is dropping high-frequency blocks at size 100 and then retaining only a few token blocks per S1. No exact compact/transliterated signature, rare-token combinations, fuzzy-within-bounded-block retrieval, explicit retrieval tiers, resume-safe candidate build signature, or per-miss diagnostics currently exist. Country is a hard namespace in the current blocking path. The Person 3 implementation will add complementary keys and measurable provenance while leaving `TextCleaner`, feature columns, model training, threshold logic, prediction, and the full-pipeline entry point intact.

## Constraints for following phases

The audit found no full candidate DB. The current S1/S2 checkpoint has not completed S2 and has no S3. Full training and test cache generation can only follow implementation and representative recall validation; it is a distinct, disk-intensive task. All generated Person 3 databases and reports will be under `D:/AmazonMLChallenge2026/artifacts/person3/`, never over the tracked 430 MB sample DB or the resumable full-run checkpoint. Recall must be measured against the full official training truth; the 100k diagnostic sample is not an accuracy proxy for the full corpus.
