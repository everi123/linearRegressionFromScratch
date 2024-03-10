import numpy as np

class LinearRegressionModel:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.W = None
        self.b = None



              
    def fit( self, X, Y ) : 
          
        # no_of_training_examples, no_of_features 
          
        self.m, self.n = X.shape 
          
        # weight initialization 
          
        self.W = np.zeros( self.n ) 
          
        self.b = 0
          
        self.X = X 
          
        self.Y = Y 
          
          
        # gradient descent learning 
                  
        for i in range( self.n_iterations) : 
              
            self.update_weights() 
              
        return self
      
    # Helper function to update weights in gradient descent 
      
    def update_weights( self ) : 
             
        Y_pred = self.predict( self.X ) 
          
        # calculate gradients   
      
        dW = - ( 2 * ( self.X.T ).dot( self.Y - Y_pred )  ) / self.m 
       
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m  
          
        # update weights 
      
        self.W = self.W - self.learning_rate * dW 
      
        self.b = self.b - self.learning_rate * db 
          
        return self
      
    # Hypothetical function  h( x )  
      
    def predict( self, X ) : 
      
        return X.dot( self.W ) + self.b 
     

