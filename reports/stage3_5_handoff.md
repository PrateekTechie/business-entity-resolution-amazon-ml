# Assigned work handoff: Stages 3–5

Scope: candidate feature extraction, model training/validation/threshold selection, and test scoring/output interfaces. Upstream preprocessing and blocking and downstream official packaging remain owned by their assigned contributors.

## Implementation

- Stage 3 exposes a bounded feature iterator. The compatibility helper refuses a pair set larger than one chunk instead of building an unbounded feature frame. The disk pipeline persists features in SQLite batches and emits `feature_report.json` and `feature_report_<split>.json` with row counts, the 62-column schema/hash, chunk size, cache state, elapsed seconds, and RSS.
- Stage 4 streams persisted numeric features into XGBoost external-memory quantile matrices; it uses entity-level modulo-5 validation, class weighting, early stopping, entity-macro F0.5 threshold selection, and a final deployment fit. It writes class counts, threshold sweep/report, validation/error analysis, feature gains, model, and model metadata.
- Stage 5 loads the saved Booster and schema/threshold metadata, scores feature chunks, persists selected pairs only, counts zero/multiple-prediction entities, and writes grouped output for every Source 1 record. It does not require candidate scores or candidate pairs in Python memory.

## Measured sample run

Scratch root: `D:/AmazonMLChallenge2026/artifacts/stage345_sample`.

## Candidate artifact inventory (2026-09-26)

Inspected SQLite contents read-only before selecting an input:

| Artifact | Records / candidates / truth / features | Decision |
|---|---:|---|
| `artifacts/cache/train.sqlite` | 4,106,821 / 0 / 0 / 0 | Not a candidate cache. It is the resumable upstream ingestion checkpoint (S1 completed, S2 checkpoint at 1,850,000). Preserved. |
| `artifacts/cache/benchmark.sqlite` | 150,000 / 3,277,137 / 124,573 / 250,000 | Benchmark data with synthetic-label contamination noted in prior run state; not valid training evidence. Not used. |
| `artifacts/benchmarks/engine_comparison/sqlite_100000/artifacts/cache/benchmark.sqlite` | 100,000 / 1,216,040 / 0 / 0 | Real upstream subset candidate set, but no truth/features; cannot train or validate by itself. |
| `artifacts/stage345_sample/artifacts/cache/train.sqlite` | 100,000 / 1,216,040 / 61,255 / 1,216,040 | Largest available suitable Stage 3–5 diagnostic cache; selected. It contains 17,616 Source 1 and 82,384 target records. |

No usable full training candidate set or test candidate set was found under `D:/AmazonMLChallenge2026/artifacts`. The top-level `artifacts/candidates`, `artifacts/features`, and `artifacts/predictions` directories contain no files. The selected sample's truth set is the official truth filtered to its Source 1 IDs; because most target records are absent from this truncated sample, its validation results are diagnostic only.

| Stage | Measured result |
|---|---|
| Feature extraction | 1,216,040 candidate pairs/features; 393.47 s (3,090 pairs/s, approximately 185,200/min); chunk size 25,000; start/end RSS 137.1/547.3 MiB; sampled peak RSS 695 MiB from the stage monitor; D: free fell from 130.64 to 130.45 GiB. |
| Training classes | 968,955 train negatives / 369 train positives; 246,621 validation negatives / 95 validation positives. |
| Validation threshold | 0.2067 on the configured 25-point grid; macro precision 0.0799, macro recall 0.0627, macro F0.5 0.0711; 122 predicted pairs among 3,535 validation entities. |
| Candidate-only CLI training | 629.5 s total; Stage 3 reused the existing 1,216,040 feature rows (0.324 s cache check); sampled peak RSS 282 MiB; no ingestion or blocking ran. |
| Candidate-only CLI scoring | 29.0 s; scored 1,216,040 pairs; 522 predicted pairs; 17,100 zero-match and 6 multi-match Source 1 records; sampled peak RSS 239 MiB. This used the training diagnostic cache as a scoring smoke input, not a held-out test cache. |

These numbers come from the generated reports under the scratch root. The benchmark upstream candidate DB has 100,000 records across sources (17,616 S1, 40,190 S2, 42,194 S3) and 1,216,040 candidate pairs. The target side is truncated relative to the full challenge corpus. Consequently these validation metrics and retrieval recall are sample/pipeline diagnostics only; they are not full-corpus quality claims. The split contains only 464 candidate-positive pairs from the subset of official ground truth mapped onto sampled Source 1 IDs (369 train and 95 validation).

For the candidate-only rerun, reports are under `D:/AmazonMLChallenge2026/artifacts/stage345_handoff/artifacts/reports`; it saved `artifacts/models/xgboost_matcher.json` plus `.metadata.json`. The successful candidate-only CLI reports 629.0 s training elapsed, 0.324 s Stage 3 cache check, and sampled peak RSS 282 MiB. Earlier diagnostics under `stage345_sample` remain intact. `performance_stage4_train_sample.json` is from an earlier failed integration attempt and is not used for the new run's measurement.

Stage 5 candidate-only scoring wrote predictions to `D:/AmazonMLChallenge2026/artifacts/stage345_prediction/submission/output/matching_results.tsv` and candidate output to `candidate_pairs.tsv`, with reports under `artifacts/reports`. This is a diagnostic over the same truncated train sample, not official test-set inference or challenge prediction statistics.

## Correctness and constraints

- Feature output order and candidate IDs are preserved by the iterator and chunked persistence; no blocking keys or candidate generation logic were changed as part of this assigned work.
- The feature schema marker was advanced to `feature-v5` so stale feature rows are not reused under the expanded 62-feature schema.
- SQLite remains the resumable feature/score spool, XGBoost matrix construction is external-memory, and prediction is chunked. The 1,216,040-row test case verifies bounded operation at sample scale, not peak full-corpus resource use.
- Synthetic tests cover streaming feature chunks, pair order/values, empty and malformed fields, the over-limit convenience guard, and class weighting. Full-corpus recall and official validation must be obtained from the integrated run and must not be inferred from this sample.

## Integrated handoff

From the project root, use the existing configured integration commands after the upstream owner has produced the training and test candidate caches:

```powershell
python main.py train --from-candidates --candidate-db D:\AmazonMLChallenge2026\artifacts\stage345_sample\artifacts\cache\train.sqlite --artifact-root D:\AmazonMLChallenge2026\artifacts\stage345_handoff --model-path D:\AmazonMLChallenge2026\artifacts\stage345_handoff\artifacts\models\xgboost_matcher.json --reuse-features
python main.py predict --from-candidates --candidate-db D:\AmazonMLChallenge2026\artifacts\stage345_sample\artifacts\cache\train.sqlite --artifact-root D:\AmazonMLChallenge2026\artifacts\stage345_prediction --model-path D:\AmazonMLChallenge2026\artifacts\stage345_handoff\artifacts\models\xgboost_matcher.json --reuse-features
```

Both commands above were executed successfully. The predict command intentionally scores the same training sample solely as a Stage 5 diagnostic; replace that path with a real test candidate cache once upstream provides one. No test candidate cache currently exists. The CLI requires an explicit existing SQLite path and fails before processing if it lacks records/candidates (and truth for training). It never guesses a cache or calls upstream ingestion/blocking. Deployment model metadata stores the exact feature column ordering and selected threshold.
