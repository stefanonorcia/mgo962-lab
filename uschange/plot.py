import pandas as pd
import matplotlib.pyplot as plt

# settings
plt.style.use('classic')
plt.rcParams["figure.figsize"] = (16, 8)

# load the data
df = pd.read_csv("uschange.csv", parse_dates=True)

print(len(df.values))

# Total
# dft = df.groupby("index")
# dft = dft['value'].agg(sum)
df.plot()
plt.show()