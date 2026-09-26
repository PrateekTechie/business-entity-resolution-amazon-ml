"""Command-line entry points for the D:-backed challenge pipeline."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict) or "storage" not in value:
        raise ValueError(f"Invalid or incomplete pipeline configuration: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "benchmark", "candidate-recall", "train", "predict", "full", "validate", "package", "person3-train", "person3-test"), nargs="?", default="full")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--reuse-candidates", action="store_true")
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--from-candidates", action="store_true",
                        help="Skip upstream ingestion/blocking and run only assigned stages on an existing SQLite candidate cache")
    parser.add_argument("--candidate-db", help="Existing SQLite database containing records and candidates (training also requires truth)")
    parser.add_argument("--artifact-root", help="Separate root for reports, model, outputs, and temporary files")
    parser.add_argument("--model-path", help="Override the model checkpoint path")
    parser.add_argument("--person3-root", help="Person 3 artifact root (defaults to <storage.root>/artifacts/person3)")
    parser.add_argument("--person3-rows-per-source", type=int, help="Optional small-sample limit for benchmark/test; omit for full corpus")
    parser.add_argument("--person3-max-s1-id", type=int, help="Optional numeric Source 1 suffix limit for diagnostics")
    parser.add_argument("--person3-chunk-rows", type=int, default=5000)
    parser.add_argument("--person3-max-block-pairs", type=int, default=250000)
    parser.add_argument("--person3-name-tokens", type=int, default=6)
    parser.add_argument("--person3-address-tokens", type=int, default=2)
    parser.add_argument("--person3-fuzzy-threshold", type=int, default=82)
    parser.add_argument("--person3-fuzzy-trigger", type=int, default=2)
    parser.add_argument("--person3-disable-families", default="",
                        help="Comma-separated retrieval family IDs to disable for controlled diagnostics")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = Path(__file__).resolve().parent
    config = load_config(str(config_path))
    storage_root = Path(config["storage"]["root"])
    if args.artifact_root:
        config.setdefault("storage", {})["root"] = str(Path(args.artifact_root).resolve())
        config["storage"]["temp_dir"] = str(Path(args.artifact_root).resolve() / "artifacts/temp")
    if args.model_path:
        config.setdefault("model", {})["checkpoint_path"] = str(Path(args.model_path).resolve())
    if args.from_candidates:
        if args.mode not in {"train", "predict"}:
            parser.error("--from-candidates is supported only with train or predict")
        if not args.candidate_db:
            parser.error("--from-candidates requires --candidate-db; no candidate cache is guessed")
    elif args.candidate_db or args.artifact_root or args.model_path:
        parser.error("--candidate-db, --artifact-root, and --model-path require --from-candidates")
    if args.mode in {"person3-train", "person3-test"}:
        if args.from_candidates or args.candidate_db or args.model_path:
            parser.error("Person 3 build modes do not accept Stage 3–5 candidate/model overrides")
        from src.blocking.person3_retrieval import Person3CandidateBuilder, generate_candidate_tsv

        person3_root = Path(args.person3_root).resolve() if args.person3_root else storage_root / "artifacts/person3"
        split = "train" if args.mode == "person3-train" else "test"
        prefix = "train" if split == "train" else "test"
        paths = {f"source{i}": config["data_paths"][f"{prefix}_source{i}"] for i in (1, 2, 3)}
        rows_limit = args.person3_rows_per_source
        if rows_limit is not None and rows_limit <= 0:
            parser.error("--person3-rows-per-source must be positive")
        builder = Person3CandidateBuilder(
            root=person3_root, mappings=project_root / "config/text_mappings.json",
            chunk_rows=args.person3_chunk_rows, max_block_pairs=args.person3_max_block_pairs,
            name_tokens_per_record=args.person3_name_tokens,
            address_tokens_per_record=args.person3_address_tokens,
            rows_per_source=rows_limit, max_s1_id=args.person3_max_s1_id,
            fuzzy_threshold=args.person3_fuzzy_threshold,
            fuzzy_trigger_candidate_count=args.person3_fuzzy_trigger,
            disabled_families=[int(value) for value in args.person3_disable_families.split(",") if value.strip()])
        database = person3_root / split / "candidate.sqlite"
        result = builder.build(split, paths, output_db=database,
                               truth_path=config["data_paths"]["train_ground_truth"] if split == "train" else None)
        if split == "test":
            candidate_path = person3_root / "test/candidate_pairs.tsv"
            output_stats = generate_candidate_tsv(database, candidate_path)
            result["candidate_tsv"] = str(candidate_path)
            result["candidate_tsv_validation"] = output_stats
            report_path = person3_root / "reports/person3_test_build.json"
            report_path.write_text(__import__("json").dumps(result, indent=2), encoding="utf-8")
            print(f"[test] candidate TSV={candidate_path}; S1 rows={output_stats['source1_rows']:,}; "
                  f"candidate pairs={output_stats['candidate_pairs']:,}", flush=True)
        return 0
    temp_dir = Path(config["storage"].get("temp_dir", storage_root / "artifacts/temp"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    tempfile.tempdir = str(temp_dir)

    from src.pipeline.disk_runner import DiskPipeline
    runner = DiskPipeline(config, project_root, force_rebuild=args.force_rebuild,
                          reuse_candidates=args.reuse_candidates, reuse_features=args.reuse_features)
    if args.mode == "smoke":
        import subprocess
        return subprocess.run([os.sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                              cwd=project_root, check=False).returncode
    if args.mode == "benchmark":
        result = runner.run("benchmark")
        print(result)
        return 0
    if args.mode == "candidate-recall":
        runner.run("candidate-recall")
        return 0
    if args.mode == "train":
        if args.from_candidates:
            runner.run_from_candidates("train", args.candidate_db)
            return 0
        runner.run("train")
        return 0
    if args.mode == "predict":
        if args.from_candidates:
            runner.run_from_candidates("test", args.candidate_db)
            return 0
        runner.run("predict")
        return 0
    if args.mode == "full":
        train_result = runner.run("train")
        print("Training complete:", train_result)
        test_result = runner.run("predict")
        print("Inference complete:", test_result)
        code = runner.validate()
        if code:
            return code
        runner.package()
        return 0
    if args.mode == "validate":
        return runner.validate()
    if args.mode == "package":
        validation = runner.root / "artifacts/reports/validator_exit_code.txt"
        if not validation.is_file() or validation.read_text(encoding="utf-8").strip() != "0":
            raise RuntimeError("Run official validator successfully before packaging")
        runner.package()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
