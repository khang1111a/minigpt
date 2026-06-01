import subprocess
import sys
from pathlib import Path
import json

def test_train_with_run_id_writes_run_outputs(tmp_path):
    config_path = tmp_path / "train_smoke_run.py"
    runs_dir = tmp_path / "runs"

    config_path.write_text(
        "\n".join(
            [
                'dataset = "tiny_text_char"',
                "batch_size = 4",
                "block_size = 8",
                "max_iters = 2",
                "eval_interval = 1",
                "eval_iters = 1",
                "learning_rate = 1e-3",
                "n_embd = 16",
                "num_heads = 4",
                "n_layer = 1",
                "dropout = 0.1",
                "seed = 42",
                'device = "cpu"',
                'out_dir = "outputs"',
                'ckpt_name = "gpt_smoke.pt"',
                'results_dir = "results"',
                'loss_name = "loss_smoke.csv"',
                "generate_after_train = False",
                "max_new_tokens = 20",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "minigpt",
            "train",
            "--config",
            str(config_path),
            "--run-id",
            "smoke-run",
            "--runs-dir",
            str(runs_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    run_dir = runs_dir / "smoke-run"

    assert "run dir:" in result.stdout
    assert (run_dir / "config.py").exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "checkpoints" / "gpt_smoke.pt").exists()
    assert (run_dir / "metrics" / "loss_smoke.csv").exists()

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["status"] == "completed"
    assert metadata["dataset"] == "tiny_text_char"
    assert metadata["device"] == "cpu"
    assert metadata["seed"] == 42
    assert metadata["max_iters"] == 2
    assert metadata["last_step"] == 1
    assert metadata["checkpoint_path"].endswith("gpt_smoke.pt")
    assert metadata["loss_path"].endswith("loss_smoke.csv")