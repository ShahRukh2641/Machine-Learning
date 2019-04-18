import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


df = pd.read_csv("data2.csv")
df

%matplotlib inline
plt.xlabel('Years')
plt.ylabel('Net Per Capita Income(US$)')
plt.scatter(df.year, df.income, color='red', marker='+')
plt.plot(df.year,reg.predict(df[['year']]), color='blue')

reg = linear_model.LinearRegression()
df['income'] = df['income'].str.replace(',', '').astype(float)
reg.fit(df[['year']], df.income)

reg.predict([[2020]])
