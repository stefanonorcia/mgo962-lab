import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import warnings

# load the data
df = pd.read_csv("tourism.csv", parse_dates=True).drop(columns=["Unnamed: 0"])

# lowercase the column names
df.columns = [col_name.lower() for col_name in df.columns]

# Purpose
purposes = df["purpose"].unique()
for purpose in purposes:
    dfp = df.query('purpose == "' + purpose + '"')
    dfp = dfp.groupby("quarter")
    dfp = dfp['trips'].agg(sum)
    dfp.plot(label=purpose)
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
plt.grid()
plt.show()
