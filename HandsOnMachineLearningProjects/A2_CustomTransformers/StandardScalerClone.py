import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class StandardScalerClone(BaseEstimator):
    def __init__(self, with_mean=True):
        self.n_features_in_ = None
        self.scale_ = None
        self.mean_ = None
        self.with_mean = with_mean

    def fit(self, x, y=None):
        X = check_array(x)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, x):
        check_is_fitted(self)
        x = check_array(x)
        assert self.n_features_in_ == x.shape[1]

        if self.with_mean:
            x = x - self.mean_

        return x / self.scale_


if __name__ == '__main__':
    array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 500])
    array = array[:, np.newaxis]
    scaler = StandardScalerClone()
    scaler.fit(array)
    print(scaler.transform(array))
