import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# List of your CSVs and their dataset names
csv_files = [
    ("Fashion-MNIST", "Fashion-MNIST-metrics.csv"),
    ("Medical-MNIST", "Medical-MNIST-metrics.csv"),
    ("MNIST", "MNIST-metrics.csv"),
]

data_dir = os.path.dirname(__file__)
data = {}
for name, fname in csv_files:
    path = os.path.join(data_dir, fname)
    df = pd.read_csv(path, index_col=0)
    data[name] = df

algorithms = data[csv_files[0][0]].index.tolist()
metrics = data[csv_files[0][0]].columns.tolist()
datasets = [name for name, _ in csv_files]

fig, axes = plt.subplots(len(datasets), len(algorithms), figsize=(4 * len(algorithms), 3 * len(datasets)))

if len(datasets) == 1 and len(algorithms) == 1:
    axes = np.array([[axes]])
elif len(datasets) == 1:
    axes = axes[np.newaxis, :]
elif len(algorithms) == 1:
    axes = axes[:, np.newaxis]

for i, dataset in enumerate(datasets):
    for j, algorithm in enumerate(algorithms):
        ax = axes[i, j]
        row = data[dataset].loc[algorithm].values.reshape(1, -1)
        sns.heatmap(row, annot=True, cbar=False, ax=ax, xticklabels=metrics, yticklabels=[], vmin=0, vmax=1, cmap="YlGnBu")
        ax.set_title(f"{dataset}\n{algorithm}")
        ax.set_xlabel("")
        ax.set_ylabel("")

plt.tight_layout()
plt.show()
