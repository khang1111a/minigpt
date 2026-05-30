import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

loss_path = ROOT / "results" / "loss.csv"
fig_path = ROOT / "results" / "loss.png"

if not loss_path.exists():
    raise FileNotFoundError(f"loss file not found: {loss_path}")

steps = []
losses = []

with open(loss_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        steps.append(int(row["step"]))
        losses.append(float(row["loss"]))

if len(steps) == 0:
    raise ValueError("loss.csv is empty")

plt.figure(figsize=(8, 5))
plt.plot(steps, losses, marker="o")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.tight_layout()

plt.savefig(fig_path, dpi=200)
plt.show()

print("loaded loss records:", len(steps))
print("saved figure:", fig_path)