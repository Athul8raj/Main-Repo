import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import mean_squared_error

df = pd.read_csv('cpu-performance.txt',na_filter=False)
#print(df.shape)
df = df[['MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP']]

X = np.array(df.drop(['ERP','PRP'],1),dtype=np.float64)
y = np.array(df['PRP'],dtype=np.float64)

X = preprocessing.scale(X)
y = preprocessing.scale(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=7)

reg = LinearRegression()
reg.fit(X_train,y_train)

accuracy = reg.score(X_test,y_test)
Y_pred = reg.predict(X_test)
print("Accuracy with normal Regression:",accuracy)
#print('Coefficient: \n', reg.coef_)
#print('Intercept: \n', reg.intercept_)
mse = mean_squared_error(y_test, Y_pred)
rmse = np.sqrt(mse)

print("RMSE with normal Regression:", rmse)