import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("PBS.csv", header=0, parse_dates=True, index_col=0)

dfs = df.query('ATC2 == "A10"')
dft = dfs.groupby('Month')
dft = dft['Cost'].agg(sum)
dft.plot()

plt.grid()
plt.show()
