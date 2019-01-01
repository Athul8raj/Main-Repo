from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score,mean_squared_error


def preprocess(X,normalize=True):
    if not normalize:
        X = preprocessing.scale(X)
    else:
        X = preprocessing.Normalizer(norm='l2').fit_transform(X) 
    return X

def label_encoder(y,one_hot=True):
    if not one_hot:
        y = preprocessing.LabelEncoder().fit(y)
    else:
        y = preprocessing.OneHotEncoder().fit(y)
    return y
        
def cross_validate(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    return X_train,X_test,y_train,y_test

    
    
class Regressor:
    def __init__(self,*regressors):
        self._regressors = regressors
        
    def train(self,X,y):
        X,y,X_test,y_test = cross_validate(X,y)
        self.clf = self._regressors.fit(X,y)
        accuracy = self.clf.score(X_test,y_test)
        co-efficients = self.clf.coef_
        intercepts = self.clf.intercept_        
        return accuracy,co-efficients,intercepts
    
    def error_rate(self,X,y):
        X,y,X_test,y_test = cross_validate(X,y)
        y_pred = self.clf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        return rmse
    

reg = Regressor(LinearRegression(),LogisticRegression())
acc,co-effs,intercepts = reg.train(X,y)
error = reg.error_rate(X,y)
        
        
        
        