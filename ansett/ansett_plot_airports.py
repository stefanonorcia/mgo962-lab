import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("ansett.csv", parse_dates=True)

# States
states = df["Airports"].unique()
for state in states:
    dfs = df.query('Airports == "' + state + '"')
    dfs = dfs.groupby("Week")
    dfs = dfs['Passengers'].agg(sum)
    dfs.plot(label=state)

plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)
plt.grid()
plt.show()
