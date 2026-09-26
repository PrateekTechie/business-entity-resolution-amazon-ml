# Person 3 candidate retrieval report

## Scope and preservation

This implementation adds an independent disk-backed candidate builder in
`src/blocking/person3_retrieval.py`. It writes only to the configured
`artifacts/person3/{train,test}` tree. It does not modify Person 2's feature,
training, prediction, or candidate-only code and does not overwrite either
existing `artifacts/cache/*.sqlite` database. The existing repository cache is
a 100,000-row-per-source diagnostic with 1,216,040 candidate pairs; the active
D: train cache is an upstream resumable ingestion cache and also remains
untouched. No test candidate cache was found during the artifact audit.

The full source sizes are 2,206,821 / 5,034,616 / 5,285,603 train rows
(S1/S2/S3) and 1,732,544 / 4,887,273 / 5,082,316 test rows. Their TSVs are in
`D:/AmazonMLChallenge2026/data/{train,test}`.

## Existing pipeline audit

The existing preprocessor lowercases and normalizes punctuation and
whitespace, expands configured terms, and leaves country values open-set.
Person 2's SQLite blocker emits six country-scoped exact name, name-token,
prefix, address-number/name-token, exact-address, and address-token blocks;
blocks above `max_group_size` are skipped. Stage 3 consumes
`records(entity_id,source_id,clean_name,clean_address,clean_country)` and
`candidates(source1_entity_id,target_entity_id,block_flags)`. It extracts the
62 existing numeric features. Stage 5 writes one row per S1, including empty
candidate lists, under the header
`source1_entity_id\tcandidate_entity_ids`.

The new database retains that Stage 3 schema, adds raw text and numeric IDs to
`records`, adds `retrieval_flags` beside the compatible `block_flags`, and
creates the same 62-column `features` table. Candidate pairs have a composite
primary key and are deduplicated across retrieval families.

## Added retrieval design

The builder streams TSV chunks and stores keys, token document frequencies,
block statistics, candidates, and provenance in SQLite. It uses deterministic
64-bit BLAKE2 keys. Families are exact normalized name, country+exact name,
compact name, Unicode NFKD ASCII folding, order-invariant informative-token
signature, corpus-rare name token, rare-token pairs, exact address, address
token, country+address number, name+address number, name+address token, and a
RapidFuzz WRatio fallback inside bounded four-character token-prefix blocks.
Country evidence is an additional key and never a hard filter.

The current defaults are a 250,000-pair maximum per key block, six selected
informative name tokens, two selected address tokens, fuzzy threshold 82, and
fuzzy fallback for S1 rows with fewer than two candidates. These defaults are
configurable. Oversized blocks are explicitly counted and their theoretical
pair products are reported; skipped blocks are diagnosed against ground truth
in the misses TSV. These values are implementation defaults, not yet a
full-corpus-validated frozen configuration.

Resume checkpoints are committed by source chunk and by candidate family/S1
ID range. The schema/checkpoint signature includes input file metadata and
retrieval settings. SQLite indexes over the largest ingested tables are now
created after the bulk data load. Python keeps one TSV chunk and bounded SQL
fetches in memory; candidate pairs remain on disk.

## Measured diagnostic

The available completed run used the first 5,000 TSV rows of each source and
the official ground truth filtered to those 5,000 S1 IDs. Its target corpus is
therefore deliberately incomplete. It is useful for exercising retrieval,
provenance, auditing, and the Stage 3 schema, but it cannot represent full
training recall or final candidate volume.

| Measure | Observed |
|---|---:|
| Indexed records | 15,000 |
| Candidate pairs | 501,232 |
| Candidates per S1 | mean 100.25; median 51; p90 280; p95 378; p99 533; max 902 |
| True pairs for indexed S1 in full GT | 17,362 |
| Those truth targets present in the 5k target sample | 24 |
| Recovered of target-present truth pairs | 24 / 24 (100%) |
| Pair recall against all full-GT truth for those S1 | 24 / 17,362 (0.1382%); invalid as a retrieval-quality estimate because targets were absent |
| Misses caused by sample target absence | 17,338 |
| Oversized blocks | 16 of 17,907; theoretical omitted products 2,983,240 |
| No-match S1 with at least one candidate | 286 / 289 (98.96%) |
| RSS at completion | 112 MiB; sampled peak 69.5 MiB (under-sampled) |
| Candidate DB size | 138,817,536 bytes |

The run had reused a previously completed candidate-family checkpoint. Its
reported 7.52-second wall time and 1.09-second candidate phase are cached
resume timings, not fresh-build throughput; they must not be used as a
performance claim. The first prototype indexed 15,000 rows in source phases
reported around 1.5–2.7 seconds each, but it was not a clean benchmark run.
No valid fresh full-corpus or representative full-target runtime has been
measured yet.

Machine-readable details and the pair-level miss file are stored with that
diagnostic under `D:/AmazonMLChallenge2026/artifacts/person3/bench_5000_v2/reports/`.
Its exact benchmark configuration used `--person3-max-block-pairs 50000`, not
the current default.

## Validation and limitations

The repository test suite passes 15 tests. Person 3 tests cover deterministic
keys, Unicode/accent folding, word-order and suffix variation, open-set
countries, S2/S3 union, deduplication, provenance, recall/miss diagnostics,
oversized blocks, Stage 3 feature schema compatibility, zero-candidate TSV
rows, checkpoint/resume, and refusal to adopt an unmarked database.

The existing 1,216,040-pair artifact was not used as a full-scale recall
benchmark: it is a diagnostic sample. The 5k prefix diagnostic has only 24
available true targets, so its 100% conditional recall has a very small
denominator and must not be presented as production-quality recall. The
candidate volume and 98.96% no-match candidate rate show that block settings
need validation on a representative target-complete sample before full
production caches are run. Transliteration is deterministic Unicode
normalization/ASCII folding, not a language-specific transliteration engine.

No full train cache, full test cache, or full-test `candidate_pairs.tsv` has
been built or validated yet. No test recall is claimed. The full cache commands
are `python main.py person3-train` and `python main.py person3-test`; they use
the D: paths in `config/config.yaml` and can resume in their separate output
directories. The test command validates one row per indexed S1, duplicate
pairs, and candidate target IDs.

## Verified 100k-per-source SQLite audit

The completed `D:/AmazonMLChallenge2026/artifacts/person3/bench_100k_v3/train/candidate.sqlite`
was checked after its builder exited. File size is 3,113,521,152 bytes; last
write time is 2026-09-26 18:38:46 local. No Python process had an open handle
to it during the audit. A normal `sqlite3.connect(path)` connection queried
the database and ran the full `PRAGMA integrity_check`, which returned `ok`
in 819.47 seconds.

| SQLite check | Result |
|---|---:|
| Total records | 300,000 |
| Retrieval-key rows | 9,223,437 |
| Candidate pairs | 20,365,018 |
| Stored training truth pairs for sampled S1 | 346,089 |
| Source 1 / Source 2 / Source 3 rows | 100,000 / 100,000 / 100,000 |
| Full integrity check | `ok` |

Recall is computed only against truth targets present in the sampled S2/S3
records. The official truth has 346,089 pairs for these S1 rows, but only 6,672
pairs (from 6,452 S1 entities) have a target in the sampled target corpus.
The retriever recovered 6,557 and missed 115 of those 6,672 available pairs:
98.2764% pair recall. It completely recovered every available pair for 6,339
of the 6,452 eligible positive S1s (98.2486% complete-match entity coverage);
6,345 (98.3416%) had at least one available true candidate. These are
conditional sample metrics, not full-corpus recall.

There are 5,548 sampled S1s with no training truth; 5,544 received at least
one candidate (99.9279%), while four received none. Across all 100,000 S1s,
candidate counts have mean 203.65, median 173, p90 416, p95 503, p99 702,
and maximum 8,103. Buckets: 0=121, 1=194, 2–5=1,259, 6–10=1,581, 11–20=2,928,
21–50=9,638, 51–100=16,368, 101–500=62,808, 501–1,000=4,988, 1,001+=115.

Per-family pair counts below overlap: a pair can have multiple provenance
bits. “Attributed candidates” counts unique candidate pairs carrying that
family's bit, not candidates exclusively owned by it.

| Retrieval family | Available true pairs recovered | Candidate pairs attributed |
|---|---:|---:|
| Exact normalized name | 1,597 | 22,408 |
| Compact name | 1,643 | 23,312 |
| Informative/rare token | 4,440 | 8,301,553 |
| Token pair | 4,310 | 2,177,153 |
| Name + address token | 4,988 | 98,657 |
| Name + address number | 4,162 | 1,456,799 |
| Address based (address token, exact address, country + address number) | 6,109 | 9,192,818 |
| Fuzzy fallback | 4 | 554 |

The 115 target-present misses are in
[`person3_bench_100k_target_present_misses.tsv`](person3_bench_100k_target_present_misses.tsv)
with raw and normalized S1/target attributes, shared key-family diagnostics,
and miss reason. The machine-readable conditional audit is
[`person3_bench_100k_audit.json`](person3_bench_100k_audit.json). The builder's
full-sample misses TSV remains under the D: benchmark report directory.

This benchmark is valid SQLite evidence and materially better recall evidence
than the 5k prefix run because the target sample provides 6,672 true pairs.
It is still not safe to call production-ready: 20.37M pairs for 100k S1s and a
99.93% no-match candidate rate indicate high downstream volume, while full
target-corpus block sizes will differ. No full train/test build should start
until the bounded block-budget comparison and candidate-volume tradeoff are
reviewed. There is no measured test-candidate count or recall because no test
candidate build has run.

The bounded comparison at a 10,000-pair block budget completed in its separate
`bench_100k_budget10k` directory using the same 100k-per-source sample. It
generated 10,312,180 pairs (103.12 per S1; median 87, p90 209, p95 251, p99
342, max 872), versus 20,365,018 at 50,000. Target-present recall was
6,526/6,672 = 97.8118%, a 0.4646 percentage-point drop; available positive
entity coverage was 97.9076%. The no-match candidate rate remained 99.9099%
(5,543/5,548). Thus lowering the block budget roughly halved candidate count
while losing 31 available true pairs; it did not solve no-match candidate
inflation. The candidate DB was 2,420,826,112 bytes. Its runtime overlapped
the first DB's integrity scan and is not treated as a comparable benchmark.

**Decision: do not start a full-scale build yet.** Both measured configurations
have very high no-match candidate rates, and the target-complete fraction is
still a sampled subset. The 10k setting trades measurable recall for lower
volume without adequately controlling false candidate load. More targeted
audit and retrieval tuning is required before selecting a full-scale setting.
