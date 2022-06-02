
from sklearn.base import BaseEstimator, TransformerMixin

from MinTrainer.tools import minkowski_distance


class DistanceTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X["distance"] = minkowski_distance(X, p=1)

        return X[["distance"]]
