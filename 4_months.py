import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("/BTC 4 month data_updated.csv")
print(df)

%matplotlib inline
plt.xlabel("Days")
plt.ylabel("Price")
plt.scatter(df["Days"], df["Adj Close"])

reg = linear_model.LinearRegression()
reg.fit(df[["Days"]], df["Adj Close"])

print(reg.predict([[121]]))
