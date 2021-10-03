import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore")

# load the data
df = pd.read_csv("tourism.csv", parse_dates=True).drop(columns=["Unnamed: 0"])

# lowercase the column names
df.columns = [col_name.lower() for col_name in df.columns]

# sum the trips over purpose
# df.groupby("state")
print(len(df.values))

# Total
dft = df.groupby("quarter")
dft = dft['trips'].agg(sum)
dft.plot()
plt.grid()
plt.show()