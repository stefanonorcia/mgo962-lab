import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("arrivals.csv", parse_dates=True)

# Total
dft = df.groupby("index")
dft = dft['value'].agg(sum)
dft.plot()
plt.grid()
plt.show()

# States
states = df["key"].unique()
for state in states:
    dfs = df.query('key == "' + state + '"')
    dfs = dfs.groupby("index")
    dfs = dfs['value'].agg(sum)
    dfs.plot(label=state)

plt.legend()
plt.grid()
plt.show()
