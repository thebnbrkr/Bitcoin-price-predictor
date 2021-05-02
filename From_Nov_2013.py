import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

pip install yfinance

import yfinance as yf

ticker = ["BTC-USD"]
p = yf.download(ticker, start="2013-11-20", end="2021-04-30")
print(p)
p.to_csv("/content/BTC complete.csv")

df = pd.read_csv("/content/BTC complete.csv")
print(df)

%matplotlib inline
plt.xlabel("Days")
plt.ylabel("Price (US$)")
plt.scatter(df["Days"], df["Adj Close"])

reg = linear_model.LinearRegression()
reg.fit(df[["Days"]], df["Adj Close"])

print(reg.predict([[2415]]))
