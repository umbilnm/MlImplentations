import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
X, y = make_regression(10000, 100)
class GradientBoostingRegressor:
    '''
    My GradientBoostingRegressor 
    '''
    def __init__(
        self,
        n_estimators=200,
        learning_rate=0.2,
        max_depth=3,
        min_samples_split=2,
        loss="mse",
        verbose=False,
        subsample_size=0.5,
        replace=False
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.verbose = verbose
        self.trees_ = []
        self.base_pred_ = None
        self.subsample_size=subsample_size
        self.replace=replace

    def _mse(self, y_true, y_pred):
        """Mean absolute error loss function and gradient."""
        loss = ((y_pred - y_true)**2).mean()
        grad = (y_pred - y_true)
        return loss, grad
    
    def _subsample(self, X, grad):
        indxs=np.random.choice(a=np.arange(0, grad.size), size=int(self.subsample_size*grad.size), replace=self.replace)
        return X[indxs, :], grad[indxs]

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        self.base_pred_ = y.mean()
        for _ in range(self.n_estimators):
            if len(self.trees_) ==0:

                loss, grad = self._mse(y, self.base_pred_)
                subsample_X, subsample_grad = self._subsample(X, grad)
                tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
                tree.fit(subsample_X, -subsample_grad)
                self.trees_.append(tree)
            else:
                current_pred = self.base_pred_ + np.array([model.predict(X) for model in self.trees_]).\
                                                sum(axis=0)*self.learning_rate        
                loss, grad = self._mse(y, current_pred)
                subsample_X, subsample_grad = self._subsample(X, grad)
                tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
                tree.fit(subsample_X, -subsample_grad)
                self.trees_.append(tree)

    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.
            
        """
        predictions = self.base_pred_ + np.array([model.predict(X) for model in self.trees_])\
                                        .sum(axis=0)*self.learning_rate
        return predictions


reg = GradientBoostingRegressor()
reg.fit(X,y)