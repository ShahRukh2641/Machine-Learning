import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

print("Dummy Variable")
df = pd.read_csv("dummy.csv")
dummy = pd.get_dummies(df['town'])

merged = pd.concat([df, dummy], axis='columns')

final = merged.drop(['town', 'west windsor'], axis='columns')

model = LinearRegression()

X = final.drop('price', axis='columns')
Y = final.price

model.fit(X,Y)

print(model.predict([[3400,0,0]]))

#print(model.score(X,Y))
print("------------------------------------------------------")

print("One Hot Encoder")
le = LabelEncoder()

dfle = df
dfle.town = le.fit_transform(dfle.town)

X = dfle[['town','area']].values
Y = dfle.price

ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray()
X = X[:,1:]
model = LinearRegression()
model.fit(X,Y)
print(model.predict([[0,0,3400]]))
