import csv
import json
import subprocess
import sys


def write_smoke_config(config_path):
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


def test_eval_writes_json_and_csv_outputs(tmp_path):
    config_path = tmp_path / "train_smoke_run.py"
    runs_dir = tmp_path / "runs"
    eval_json_path = tmp_path / "eval.json"
    eval_csv_path = tmp_path / "eval.csv"

    write_smoke_config(config_path)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "minigpt",
            "train",
            "--config",
            str(config_path),
            "--run-id",
            "eval-smoke",
            "--runs-dir",
            str(runs_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    ckpt_path = runs_dir / "eval-smoke" / "checkpoints" / "gpt_smoke.pt"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "minigpt",
            "eval",
            "--config",
            str(config_path),
            "--ckpt",
            str(ckpt_path),
            "--split",
            "train",
            "--eval-iters",
            "1",
            "--out",
            str(eval_json_path),
            "--csv",
            str(eval_csv_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "saved eval json:" in result.stdout
    assert "saved eval csv:" in result.stdout

    eval_json = json.loads(eval_json_path.read_text(encoding="utf-8"))

    assert eval_json["checkpoint"] == str(ckpt_path)
    assert eval_json["dataset"] == "tiny_text_char"
    assert eval_json["device"] == "cpu"
    assert eval_json["eval_iters"] == 1
    assert set(eval_json["splits"]) == {"train"}
    assert eval_json["splits"]["train"]["loss"] is not None
    assert eval_json["splits"]["train"]["perplexity"] is not None

    with open(eval_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["checkpoint"] == str(ckpt_path)
    assert rows[0]["dataset"] == "tiny_text_char"
    assert rows[0]["device"] == "cpu"
    assert rows[0]["eval_iters"] == "1"
    assert rows[0]["split"] == "train"
    assert rows[0]["loss"] != ""
    assert rows[0]["perplexity"] != ""
