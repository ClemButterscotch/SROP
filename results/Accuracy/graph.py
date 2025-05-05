import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Paths to your CSVs
files = {
    'MNIST':       'MNIST-metrics.csv',
    'Fashion-MNIST': 'Fashion-MNIST-metrics.csv',
    'Medical-MNIST': 'Medical-MNIST-metrics.csv'
}

# 2) Load & normalize
data = {}
for name, path in files.items():
    df = pd.read_csv(path).set_index('Algorithm')
    # if values are percentages (0–100), convert to [0,1]
    if df[['TestAcc','NMI','ARI']].values.max() > 1:
        df[['TestAcc','NMI','ARI']] /= 100
    data[name] = df[['TestAcc','NMI','ARI']]

# 3) Build the RGB array
algorithms = list(data[next(iter(data))].index)
datasets   = list(files.keys())
rgb = np.zeros((len(algorithms), len(datasets), 3))
for i, alg in enumerate(algorithms):
    for j, ds in enumerate(datasets):
        acc, nmi, ari = data[ds].loc[alg]
        rgb[i, j] = [1-acc, 1-nmi, 1-ari]

# 4) Plot it
fig, ax = plt.subplots(figsize=(6,5))
ax.imshow(rgb, aspect='equal')

# Labels
ax.set_xticks(np.arange(len(datasets)))
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.set_yticks(np.arange(len(algorithms)))
ax.set_yticklabels(algorithms)

# White grid lines between cells
ax.set_xticks(np.arange(-.5, len(datasets), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(algorithms), 1), minor=True)
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
ax.tick_params(which='minor', length=0)

plt.title('Clustering Metrics as RGB\n(R=Accuracy, G=NMI, B=ARI)')
plt.tight_layout()
plt.show()