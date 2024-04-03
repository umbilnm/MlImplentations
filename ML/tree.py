from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Node:
    """Decision tree node."""
    feature: int
    # YOUR CODE HERE: add the required attributes
    feature: int = None
    threshold: float = None
    n_samples: int = None
    value: float = None
    mse: float = None
    left: Node = None
    right: Node = None

@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor."""
    max_depth: int
    min_samples_split: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y)."""
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def _mse(self, y: np.ndarray) -> float:
        pred = y.mean()

        return ((y - pred)**2).mean()
    
    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weithed mse criterion for a two given sets of target values"""
        pred_left = y_left.mean()
        pred_right = y_right.mean()
        return (((y_left - pred_left)**2).mean()*y_left.size + ((y_right - pred_right)**2).mean()*y_right.size)\
            /(y_left.size + y_right.size)
    

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node."""
        best_mse = None
        best_feature = None 
        for feature in range(X.shape[1]):
            X_current = X[:, feature]
            for threshold in np.unique(X_current):
                y_left = y[X_current>threshold]
                y_right = y[X_current<=threshold]
                current_mse = self._weighted_mse(y_left, y_right)
                if best_mse is None or current_mse<best_mse:
                    best_mse = current_mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""
        current_mse = self._mse(y)
        value_ = y.mean()
        if depth == self.max_depth or X.shape[0] <= self.min_samples_split:
            node = Node(
            feature=None,
            threshold=None,
            n_samples=X.shape[0],
            value=round(value_),
            mse=current_mse,
            left=None,
            right=None
        )
            return node
        else:
            best_feature, best_threshold = self._best_split(X ,y)
            X_left = X[X[:, best_feature] <= best_threshold, :]
            y_left = y[X[:, best_feature] <= best_threshold]
            X_right = X[X[:, best_feature] > best_threshold, :]
            y_right = y[X[:, best_feature] > best_threshold]

            node = Node(
                feature=best_feature,
                threshold=best_threshold,
                n_samples=X.shape[0],
                value=round(value_),
                mse=current_mse,
                left=self._split_node(X=X_left,y=y_left,depth=depth+1), 
                right=self._split_node(X=X_right,y=y_right,depth=depth+1)
                             )
            return node
    

    def as_json(self) -> str:
        """Return the decision tree as a JSON string."""
        self.json = self._as_json(self.tree_)
        return self.json
    def _as_json(self, node: Node) -> str:
        """Return the decision tree as a JSON string. Execute recursively."""
        if node.left is None and node.right is None:
            return f'{{"value": {node.value}, "n_samples": {node.n_samples}, "mse": {round(node.mse,2)}}}'
        else:
            return f'{{"feature": {node.feature}, "threshold": {node.threshold}, "n_samples": {node.n_samples}, "mse": {round(node.mse,2)}, "left": {self._as_json(node.left)}, "right": {self._as_json(node.right)}}}'


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values. 
        """

        y = np.array([self._predict_one_sample(sample) for sample in X])
        
        return y


    def _predict_one_sample(self, features: np.ndarray) -> int:
        """Predict the target value of a single sample."""
        node = self.tree_
        while node.left is not None:
            i_feature = node.feature
            threshold = node.threshold
            if features[i_feature]<=threshold:
                node = node.left
            else:
                node = node.right

        return node.value

