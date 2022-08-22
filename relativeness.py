import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
from mlxtend.plotting import heatmap
import numpy as np

df = pd.read_csv('hydroparam.csv', header=None)
df.columns = ['B_tip', 'B_70', 'A_tip', 'A_70', 'Cp']
print(df.head())

cols = ['B_tip', 'B_70', 'A_tip', 'A_70', 'Cp']
scatterplotmatrix(df[cols].values, figsize=(10, 8), names=cols, alpha=0.5)
plt.tight_layout()
plt.show()

cm = np.corrcoef(df[cols].values.T)
hn = heatmap(cm, row_names=cols, column_names= cols)
plt.show()