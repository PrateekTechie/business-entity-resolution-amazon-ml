# Amazon ML Challenge 2026: Business Entity Resolution

The pipeline retrieves zero or more matching Source 2 and Source 3 records for every Source 1 business, using challenge-provided data only. It generates the official matching and candidate TSVs and runs the recovered official validator.

## Storage and data

The active configuration uses `D:/AmazonMLChallenge2026` for the copied train/test TSVs, SQLite staging databases, candidate and feature caches, model, reports, temporary files, outputs, and ZIP. The original challenge resource is preserved at `C:/Users/HP/Downloads/6ab10eb3b23ba_student_resource/student_resource`; the validator and documentation are copied to `D:/AmazonMLChallenge2026/challenge_resource`. Change paths in `config/config.yaml` when using a different machine. `main.py` directs Python temporary files to the configured D: temp directory.

## Pipeline

1. Stream TSV input in chunks and normalize names, addresses, and country values using the supplied mapping file. Country values remain open-set.
2. Index records and blocking keys in SQLite. Retrieve a deduplicated union of exact normalized name, dynamically ranked informative name tokens, name prefixes, address number plus name token, exact normalized address, and informative address tokens, each scoped by country. `max_group_size` bounds both sides of a block. There is no per-entity top-K truncation.
3. Join candidate records in feature-sized chunks and compute 62 numeric RapidFuzz, token, address, cross-field, and block features. IDs and text are excluded from model inputs. Stage 3 writes split-specific reports with persisted rows, feature schema hash, chunk size, elapsed time, and RSS.
4. Label retrieved training pairs from the supplied ground truth. Hold out Source 1 entities with numeric ID suffix divisible by five. Train an XGBoost histogram model with class weighting and early stopping; choose a threshold using held-out entity macro F0.5, then fit a deployment model on all training candidates.
5. Score every retrieved test pair and stream grouped output. Every test Source 1 ID receives a row; an entity can have zero or multiple matches. Candidate output is the same deduplicated set scored by the model.

The SQLite database is the resumable spool for ingestion, blocking, labels, candidates, features, validation scores, and predictions. Work is chunked; the feature matrix is numeric and XGBoost uses external-memory quantile matrices. Stage reports record elapsed time, sampled RSS, and observed D: free space. Sampled RSS is not an absolute peak guarantee.

## Commands

Run from the repository root after installing `requirements.txt`:

```powershell
python main.py smoke
python main.py benchmark
python main.py candidate-recall
python main.py train
python main.py predict
python main.py validate
python main.py package
```

For an existing integrated SQLite cache that already contains candidates, skip upstream ingestion and blocking with explicit candidate-only commands:

```powershell
python main.py train --from-candidates --candidate-db D:\AmazonMLChallenge2026\artifacts\stage345_sample\artifacts\cache\train.sqlite --artifact-root D:\AmazonMLChallenge2026\artifacts\stage345_handoff --model-path D:\AmazonMLChallenge2026\artifacts\stage345_handoff\artifacts\models\xgboost_matcher.json --reuse-features
python main.py predict --from-candidates --candidate-db D:\AmazonMLChallenge2026\artifacts\<test-candidate-cache>.sqlite --artifact-root D:\AmazonMLChallenge2026\artifacts\stage345_prediction --model-path D:\AmazonMLChallenge2026\artifacts\stage345_handoff\artifacts\models\xgboost_matcher.json --reuse-features
```

Candidate-only mode requires an explicit existing SQLite DB. Training requires its `records`, `candidates`, and `truth` tables. Prediction requires `records` and `candidates`; neither command silently falls back to upstream ingestion. Current artifact availability and sample-only diagnostic limitations are listed in `reports/stage3_5_handoff.md`.

`python main.py full` trains, predicts, runs the official validator, and packages only if validation succeeds. `--force-rebuild` rebuilds a mode's SQLite cache. Train and predict caches are separate. `--reuse-candidates` and `--reuse-features` remain supported CLI options.

## Official outputs

`D:/AmazonMLChallenge2026/submission/output/matching_results.tsv` has columns `source1_entity_id` and `matched_entity_ids`; `candidate_pairs.tsv` has `source1_entity_id` and `candidate_entity_ids`. Files are tab-separated, with comma-separated sorted IDs and a blank field for an empty set. Run the official validator with `python main.py validate`. The packaged archive is `D:/AmazonMLChallenge2026/submission/Tech_Giants_submission.zip`.

## Measurement status

The source row counts are 2,206,821 / 5,034,616 / 5,285,603 for train S1/S2/S3 and 1,732,544 / 4,887,273 / 5,082,316 for test S1/S2/S3. The configured blocking policy is `max_group_size: 100`; recall and candidate distribution must be read from the measured report at `D:/AmazonMLChallenge2026/artifacts/reports/candidate_recall_report.json`. Validation metrics, model counts, test statistics, and validator status are reported under the same D: reports directory after their respective stages complete. No full-run metric is asserted here unless a generated report supports it.

The pipeline has passed the repository's ten synthetic unit tests. Full-data retrieval, validation, inference, official validation, and packaging status are determined by their generated reports and artifacts, not by the synthetic tests. Stage 3–5 sample evidence and limitations are documented in `reports/stage3_5_handoff.md`.
