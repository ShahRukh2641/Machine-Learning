import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model


df = pd.read_csv("homeprices.csv")
df['income'] = df['income'].str.replace(',', '').astype(float)

df

median_bedroom = math.floor(df.bedrooms.median())
median_bedroom
df.bedrooms = df.bedrooms.fillna(median_bedroom)

df
reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']], df.price)

reg.predict([[3000, 3, 40]])
