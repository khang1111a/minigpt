import json
import shutil
from datetime import datetime
from pathlib import Path


def make_run_id(now=None):
    if now is None:
        now = datetime.now()

    return now.strftime("%Y%m%d-%H%M%S")


def create_run_dir(runs_dir="runs", run_id=None):
    runs_dir = Path(runs_dir)

    if run_id is None:
        run_id = make_run_id()

    run_dir = runs_dir / run_id

    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")

    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "metrics").mkdir()
    (run_dir / "samples").mkdir()

    return run_dir


def copy_config(config_path, run_dir):
    config_path = Path(config_path)
    run_dir = Path(run_dir)

    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    target_path = run_dir / "config.py"
    shutil.copy2(config_path, target_path)

    return target_path


def write_metadata(run_dir, metadata):
    run_dir = Path(run_dir)
    metadata_path = run_dir / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return metadata_path


def init_run(config_path, runs_dir="runs", run_id=None, metadata=None):
    run_dir = create_run_dir(runs_dir=runs_dir, run_id=run_id)
    copied_config = copy_config(config_path, run_dir)

    metadata = {} if metadata is None else dict(metadata)
    metadata.setdefault("run_id", run_dir.name)
    metadata.setdefault("config_path", str(Path(config_path)))
    metadata.setdefault("config_snapshot", str(copied_config))

    metadata_path = write_metadata(run_dir, metadata)

    return {
        "run_dir": run_dir,
        "config_path": copied_config,
        "metadata_path": metadata_path,
        "checkpoints_dir": run_dir / "checkpoints",
        "metrics_dir": run_dir / "metrics",
        "samples_dir": run_dir / "samples",
    }

def read_metadata(run_dir):
    run_dir = Path(run_dir)
    metadata_path = run_dir / "metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_metadata(run_dir, updates):
    metadata = read_metadata(run_dir)
    metadata.update(updates)
    return write_metadata(run_dir, metadata)