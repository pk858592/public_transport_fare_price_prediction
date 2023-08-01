import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv("dj.csv")


X = df.drop(['key','pickup_datetime','pickup_weekday','fare_amount'],axis=1)
# # dependent variable
y = df['fare_amount']

X_train, x_test, y_train, y_test = train_test_split(X, y,train_size=0.8, random_state=12345)


rf = RandomForestRegressor(n_estimators=100, max_depth=11, random_state=42)
t = rf.fit(X_train, y_train)

import pickle
pickle.dump(t,open("model.pkl","wb"))
# type(df)





