import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("ansett.csv", parse_dates=True)

print(len(df.values))

# Total
dft = df.groupby("Week")
dft = dft['Passengers'].agg(sum)
dft.plot()
plt.show()