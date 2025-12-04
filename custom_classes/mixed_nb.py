import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB, CategoricalNB

class MixedNB(BaseEstimator, ClassifierMixin):
    """
    A custom Naive Bayes classifier combining GaussianNB for numerical features 
    and CategoricalNB for categorical features, suitable for mixed data.
    """
    
    _estimator_type = "classifier"
    
    def __init__(self, var_smoothing=1e-9, alpha=1.0, num_features_count=7):
        """
        :param var_smoothing: Smoothing parameter for GaussianNB (numerical part).
        :param alpha: Smoothing parameter for CategoricalNB (categorical part).
        :param num_features_count: The number of numerical features (columns) 
                                   outputted by the preprocessing step.
        """
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.gnb = GaussianNB(var_smoothing=var_smoothing)
        self.cnb = CategoricalNB(alpha=alpha)
        self.num_features_count = num_features_count

    def fit(self, X, y):
        X_dense = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        self.gnb.fit(X_dense[:, :self.num_features_count], y)
        self.cnb.fit(X_dense[:, self.num_features_count:], y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.class_log_prior_ = np.log(counts / len(y))
        return self

    def predict_log_proba(self, X):
        X_dense = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        X_num = X_dense[:, :self.num_features_count]
        X_cat = X_dense[:, self.num_features_count:]
        return self.gnb.predict_log_proba(X_num) + self.cnb.predict_log_proba(X_cat) - self.class_log_prior_

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))