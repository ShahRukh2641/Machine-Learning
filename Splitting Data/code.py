import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("dummy.csv")

#plt.scatter(df['Mileage'], df['Sell Price($)'])

#plt.scatter(df['Age(yrs)'], df['Sell Price($)'])

X = df[['Mileage', 'Age(yrs)']]
Y = df['Sell Price($)']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

clf = LinearRegression()

clf.fit(X_train, Y_train)

print(clf.predict(X_test))
print(Y_test)

print(clf.score(X_test, Y_test))
