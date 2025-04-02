import numpy as np

class Logistic_Regression:

    # define __init__ with 2 parameter
    def __init__(self, learning_rate=0.0001, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.params = {}

    # define params_init for declare W (weight) of input and b (bias)
    def params_init(self, X):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        
        _, n_features = X.shape

        self.params = {
            'W': np.zeros(n_features),
            'b': 0
        }

        return self

    # define get_z with 3 parameter, this function count mx + b and use in sigmoid as z
    def get_z(self, X, W, b):
        z = np.dot(X, W) + b
        return z

    # define get_sigmoid with z as input, and returning sigmoid value over a range 0 and 1
    def get_sigmoid(self, z):
        g = 1 / (1 + np.exp(-z))
        return g

    ''' define gradient_descent with 2 parameter, this function reduce weight and bias 
        in every iteration, counting log-loss as cost or loss in X training, by reducing 
        w and b loss with less than before. 
    '''
    def gradient_descent(self, X_train, y_train):
        X_train = X_train.reshape(-1, 1) if X_train.ndim == 1 else X_train

        W = self.params['W']
        b = self.params['b']
        m = X_train.shape[0]
        epsilon = 1e-10

        for _ in range(self.n_iter):
            # h(xi)
            g = self.get_sigmoid(self.get_z(X_train, W, b))
            g = np.clip(g, epsilon, 1 - epsilon)
            # ngitung Log-loss (Binary Cross Entropy)
            loss = -1/m * np.sum(y_train * np.log(g) + (1 - y_train) * np.log(1 - g))
            dw = 1/m * np.dot(X_train.T, (g - y_train))
            db = 1/m * np.sum(g - y_train)
            
            W -= self.learning_rate * dw
            b -= self.learning_rate * db

        self.params['W'] = W
        self.params['b'] = b
        return self

    # define train function with 2 parameter
    def train(self, X_train, y_train):
        self.params_init(X_train)
        self.gradient_descent(X_train, y_train)
        return self

    # define predict funtion with 1 paramter
    def predict(self, X_test):
        g = self.get_sigmoid(np.dot(X_test, self.params['W']) + self.params['b'])
        return (g >= 0.5).astype(int)