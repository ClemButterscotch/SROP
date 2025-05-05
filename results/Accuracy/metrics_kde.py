import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# List of your CSVs and their dataset names
csv_files = [
    ("Fashion-MNIST", "Fashion-MNIST-metrics.csv"),
    ("Medical-MNIST", "Medical-MNIST-metrics.csv"),
    ("MNIST", "MNIST-metrics.csv"),
]

data_dir = os.path.dirname(__file__)
frames = []
for dataset, fname in csv_files:
    path = os.path.join(data_dir, fname)
    df = pd.read_csv(path)
    df['Dataset'] = dataset
    frames.append(df)

# Combine all into one DataFrame
all_metrics = pd.concat(frames, ignore_index=True)

# We'll plot TestAcc vs NMI for all algorithms and datasets
sns.kdeplot(
    data=all_metrics, x="TestAcc", y="NMI",
    fill=True, thresh=0, levels=100, cmap="mako"
)
plt.title("Bivariate KDE: TestAcc vs NMI (All Datasets/Algorithms)")
plt.show()
