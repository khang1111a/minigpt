import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--loss",
        type=str,
        default="results/loss.csv",
        help="Path to loss csv file.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to output figure.",
    )

    args = parser.parse_args()

    loss_path = Path(args.loss)

    if not loss_path.is_absolute():
        loss_path = ROOT / loss_path

    if args.out is None:
        fig_path = loss_path.with_suffix(".png")
    else:
        fig_path = Path(args.out)
        if not fig_path.is_absolute():
            fig_path = ROOT / fig_path

    if not loss_path.exists():
        raise FileNotFoundError(f"loss file not found: {loss_path}")

    steps = []
    train_losses = []
    val_losses = []

    with open(loss_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames

        if fieldnames is None:
            raise ValueError("loss.csv has no header")

        for row in reader:
            steps.append(int(row["step"]))

            if "train_loss" in fieldnames:
                train_losses.append(float(row["train_loss"]))

                val_value = row.get("val_loss", "")
                if val_value == "":
                    val_losses.append(None)
                else:
                    val_losses.append(float(val_value))

            elif "loss" in fieldnames:
                train_losses.append(float(row["loss"]))
                val_losses.append(None)

            else:
                raise ValueError(
                    "loss csv must contain either 'loss' or 'train_loss' column"
                )

    if len(steps) == 0:
        raise ValueError("loss csv is empty")

    plt.figure(figsize=(8, 5))

    plt.plot(
        steps,
        train_losses,
        marker="o",
        label="train loss",
    )

    valid_val_steps = []
    valid_val_losses = []

    for step, val_loss in zip(steps, val_losses):
        if val_loss is not None:
            valid_val_steps.append(step)
            valid_val_losses.append(val_loss)

    if len(valid_val_losses) > 0:
        plt.plot(
            valid_val_steps,
            valid_val_losses,
            marker="o",
            label="val loss",
        )

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200)
    plt.show()

    print("loss file:", loss_path)
    print("loaded loss records:", len(steps))
    print("saved figure:", fig_path)


if __name__ == "__main__":
    main()