import json

import pytest

from minigpt.runs import create_run_dir, init_run, update_metadata


def test_init_run_creates_expected_structure(tmp_path):
    config_path = tmp_path / "train_config.py"
    config_path.write_text('dataset = "tiny_text_char"\n', encoding="utf-8")

    result = init_run(
        config_path=config_path,
        runs_dir=tmp_path / "runs",
        run_id="test-run",
        metadata={
            "seed": 42,
            "device": "cpu",
        },
    )

    run_dir = result["run_dir"]

    assert run_dir.exists()
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "metrics").is_dir()
    assert (run_dir / "samples").is_dir()

    copied_config = run_dir / "config.py"
    assert copied_config.read_text(encoding="utf-8") == 'dataset = "tiny_text_char"\n'

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == "test-run"
    assert metadata["seed"] == 42
    assert metadata["device"] == "cpu"
    assert metadata["config_snapshot"] == str(copied_config)


def test_create_run_dir_rejects_existing_run(tmp_path):
    runs_dir = tmp_path / "runs"

    create_run_dir(runs_dir=runs_dir, run_id="duplicate")

    with pytest.raises(FileExistsError):
        create_run_dir(runs_dir=runs_dir, run_id="duplicate")

def test_update_metadata_merges_existing_values(tmp_path):
    config_path = tmp_path / "train_config.py"
    config_path.write_text('dataset = "tiny_text_char"\n', encoding="utf-8")

    result = init_run(
        config_path=config_path,
        runs_dir=tmp_path / "runs",
        run_id="test-run",
        metadata={
            "seed": 42,
        },
    )

    update_metadata(
        result["run_dir"],
        {
            "status": "completed",
            "last_step": 1,
        },
    )

    metadata = json.loads(
        (result["run_dir"] / "metadata.json").read_text(encoding="utf-8")
    )

    assert metadata["run_id"] == "test-run"
    assert metadata["seed"] == 42
    assert metadata["status"] == "completed"
    assert metadata["last_step"] == 1