import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ausbeer.csv", header=0, parse_dates=True, index_col=0)

df.plot()
plt.grid()
plt.show()