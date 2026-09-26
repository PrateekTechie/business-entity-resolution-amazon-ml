# Amazon ML Challenge 2026: Business Entity Resolution

**Team:** Tech Giants  
**Members:** Not provided  
**Submission date:** 2026-09-25

## Approach

The system returns zero or more Source 2 and Source 3 records matching each Source 1 record. It uses only the supplied challenge data. A country-aware, multi-rule retrieval stage feeds a binary pair classifier. The complete deduplicated retrieval set is retained; there is no per-entity candidate top-K cap.

## Data and preprocessing

The input copies are stored under `D:/AmazonMLChallenge2026/data`. Observed row counts are train S1 2,206,821, S2 5,034,616, S3 5,285,603; test S1 1,732,544, S2 4,887,273, S3 5,082,316. Names, addresses, and country labels are normalized using the challenge mapping file. Country is a string feature and blocking namespace, so unseen test values such as France remain supported. Original official source files are preserved.

## Retrieval

Records and blocking keys are indexed in a D: backed SQLite database. The candidate set unions country-scoped exact normalized name, rare informative name tokens, name prefixes, address number plus informative name token, exact normalized address, and informative address token blocks. Generic organization/address tokens are excluded from informative keys. Key selection uses observed block frequencies; oversized groups are skipped using configured `blocking.max_group_size`. Candidates are deduplicated by Source 1 and target ID in SQLite while block provenance is held as compact flags.

Full training candidate recall and the per-entity candidate distribution are measured against `train_ground_truth.tsv` and written to `D:/AmazonMLChallenge2026/artifacts/reports/candidate_recall_report.json` and `.txt`. The measured values should be reported from those files after the candidate-recall stage finishes; no estimate is substituted for the report.

## Features and model

Candidate features are calculated in chunks. The current numeric schema contains 62 name, address, cross-field, country, block-provenance, quality, and source-indicator features. Model inputs contain no IDs or raw text. Feature columns are ordered and stored with the model metadata. Non-finite and missing feature values are made finite before persistence. Stage 3 reports persisted row counts, schema hash, chunk size, runtime, RSS, and cache reuse.

Ground-truth pairs label retrieved candidates as positives; other retrieved pairs are negatives. Validation holds out complete Source 1 entities based on numeric entity ID suffix modulo five. XGBoost histogram trees use positive-class weighting and early stopping. The decision threshold is selected on held-out Source 1 entities by macro F0.5; the deployment model is refit on all training candidate pairs using the selected number of rounds.

## Evaluation

The primary metric is the official per-Source-1 macro F0.5, with correctly empty prediction and truth sets scoring 1.0. The threshold sweep records macro precision, macro recall, macro F0.5, predicted pairs, and entities with zero or nonzero predictions. Error analysis distinguishes true pairs absent from retrieval from retrieved positives rejected by the classifier. Results are written under `D:/AmazonMLChallenge2026/artifacts/reports` after training completes.

## Scalability and reproducibility

The configurable storage root is `D:/AmazonMLChallenge2026`. TSVs are ingested in chunks; SQLite stores normalized records, keys, candidates, labels, features, and predictions. Feature extraction and prediction use bounded chunks, and XGBoost training uses external-memory quantile matrices. Temporary files are directed to D:. Stage reports sample RSS and D: free space once per second; RSS is therefore an observed sampled peak, not a guaranteed absolute peak. SQLite caches allow completed ingestion, candidate, and feature stages to be reused. The official input data are not modified.

Commands from the repository root:

```powershell
python main.py smoke
python main.py benchmark
python main.py candidate-recall
python main.py train
python main.py predict
python main.py validate
python main.py package
```

The full mode performs training and test inference, then validates and packages only after the official validator succeeds. Exact measured candidate recall, training class counts, validation metrics and threshold, test candidate/match counts, stage runtime/RSS, validator result, and archive path belong here after those stages have completed. Synthetic tests alone do not establish full-scale performance.

## Limitations

Retrieval can miss matches whose supplied attributes do not share any retained block key. The configured group-size limit trades off retrieval volume and recall; the held-out/full-train recall report is required to assess that tradeoff. The learned threshold and scores are limited by the provided labels and the deterministic entity split. No performance rank is claimed.
