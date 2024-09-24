import numpy as np


class LinearRegressor:
    def __init__(self,n_iteration=1000,learning_rate=0.01) -> None:
        self.alpha=learning_rate
        self.n_iteration=n_iteration
        self.W=None
    def fit(self,X,y):
        #adding bias dimension
        X=np.concatenate(((np.ones((X.shape[0],1))),X),axis=1)
        y=np.reshape(y,(X.shape[0],1))

        #weight initialization

        self.W=np.zeros((X.shape[0],1))+0.4

        # Gradient descent
        for _ in range(self.n_iteration):
            #get the gradient
            grad_w=self.grad(X,y)
            #update the weight
            self.W-=self.alpha*grad_w


    def predict(self,X):
        return np.dot(X,self.W)
    def grad(self,X,y):
        return np.dot(X.T,(self.predict(X)-y))






     

