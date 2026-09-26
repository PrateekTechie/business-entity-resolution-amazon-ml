"""Candidate recall diagnostics against challenge-provided training labels."""
import json
import os
import numpy as np
import pandas as pd


def evaluate_candidate_recall(candidates: pd.DataFrame, ground_truth_tsv: str,
                              report_path: str = "reports/candidate_recall_report.json",
                              source1: pd.DataFrame | None = None) -> dict:
    gt = pd.read_csv(ground_truth_tsv, sep="\t", dtype=str, keep_default_na=False)
    true_pairs = []
    for s1, raw in zip(gt.source1_entity_id, gt.matched_entity_ids, strict=True):
        true_pairs.extend((s1, value.strip()) for value in raw.split(",") if value.strip())
    truth = pd.DataFrame(true_pairs, columns=["source1_entity_id", "target_entity_id"]).drop_duplicates()
    found = truth.merge(candidates.drop_duplicates(), on=["source1_entity_id", "target_entity_id"], how="inner")
    per_entity = found.groupby("source1_entity_id").size()
    expected = truth.groupby("source1_entity_id").size()
    entity_hit = per_entity.reindex(expected.index, fill_value=0)
    counts = candidates.groupby("source1_entity_id").size().reindex(gt.source1_entity_id, fill_value=0).to_numpy()
    report = {
        "true_match_pairs": int(len(truth)), "retrieved_true_match_pairs": int(len(found)),
        "candidate_recall": float(len(found) / max(1, len(truth))),
        "all_match_entity_coverage": float((entity_hit == expected).mean()) if len(expected) else 0.0,
        "zero_retrieved_true_match_entities": int((entity_hit == 0).sum()),
        "false_candidate_pairs": int(len(candidates) - len(found)),
        "source_recall": {},
        "candidate_pairs": int(len(candidates)),
        "candidates_per_entity": {"mean": float(counts.mean()) if len(counts) else 0.0,
            "median": float(np.median(counts)) if len(counts) else 0.0,
            "p90": float(np.percentile(counts, 90)) if len(counts) else 0.0,
            "p95": float(np.percentile(counts, 95)) if len(counts) else 0.0,
            "p99": float(np.percentile(counts, 99)) if len(counts) else 0.0,
            "max": int(counts.max()) if len(counts) else 0},
    }
    for source in ("S2-", "S3-"):
        subset = truth[truth.target_entity_id.str.startswith(source)]
        got = found[found.target_entity_id.str.startswith(source)]
        report["source_recall"][source.rstrip("-")] = float(len(got) / max(1, len(subset)))
    if source1 is not None:
        countries = source1[["entity_id", "clean_country"]].rename(
            columns={"entity_id": "source1_entity_id", "clean_country": "country"})
        by_country = truth.merge(countries, on="source1_entity_id", how="left").groupby("country").size()
        got_country = found.merge(countries, on="source1_entity_id", how="left").groupby("country").size()
        report["country_recall"] = {str(country): float(got_country.get(country, 0) / total)
                                     for country, total in by_country.items()}
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    text_path = os.path.splitext(report_path)[0] + ".txt"
    with open(text_path, "w", encoding="utf-8") as stream:
        stream.write("Candidate Recall Report\n")
        stream.write(f"True match pairs: {report['true_match_pairs']:,}\n")
        stream.write(f"Retrieved true match pairs: {report['retrieved_true_match_pairs']:,}\n")
        stream.write(f"Overall candidate recall: {report['candidate_recall']:.4%}\n")
        stream.write(f"All-match entity coverage: {report['all_match_entity_coverage']:.4%}\n")
        stream.write(f"Positive-match entities with zero retrieved matches: {report['zero_retrieved_true_match_entities']:,}\n")
        stream.write(f"Candidate pairs: {report['candidate_pairs']:,}\n")
        stream.write(f"False candidate pairs: {report['false_candidate_pairs']:,}\n")
        stream.write("Candidates per Source 1 entity: " + json.dumps(report["candidates_per_entity"]) + "\n")
        stream.write("Source recall: " + json.dumps(report["source_recall"]) + "\n")
        if "country_recall" in report:
            stream.write("Country recall: " + json.dumps(report["country_recall"]) + "\n")
    print(f"Candidate recall: {report['candidate_recall']:.2%}; all-match entity coverage: {report['all_match_entity_coverage']:.2%}; mean candidates/entity: {report['candidates_per_entity']['mean']:.2f}; P95: {report['candidates_per_entity']['p95']:.0f}")
    return report
