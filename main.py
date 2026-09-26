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
    parser.add_argument("mode", choices=("smoke", "benchmark", "candidate-recall", "train", "predict", "full", "validate", "package"), nargs="?", default="full")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--reuse-candidates", action="store_true")
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--from-candidates", action="store_true",
                        help="Skip upstream ingestion/blocking and run only assigned stages on an existing SQLite candidate cache")
    parser.add_argument("--candidate-db", help="Existing SQLite database containing records and candidates (training also requires truth)")
    parser.add_argument("--artifact-root", help="Separate root for reports, model, outputs, and temporary files")
    parser.add_argument("--model-path", help="Override the model checkpoint path")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = Path(__file__).resolve().parent
    config = load_config(str(config_path))
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
    storage_root = Path(config["storage"]["root"])
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
