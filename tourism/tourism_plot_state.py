import pandas as pd
import matplotlib.pyplot as plt

import warnings

# load the data
df = pd.read_csv("tourism.csv", parse_dates=True).drop(columns=["Unnamed: 0"])

# lowercase the column names
df.columns = [col_name.lower() for col_name in df.columns]

# States
states = df["state"].unique()
for state in states:
    dfs = df.query('state == "' + state + '"')
    dfs = dfs.groupby("quarter")
    dfs = dfs['trips'].agg(sum)
    dfs.plot(label=state)

plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
plt.grid()
plt.show()
