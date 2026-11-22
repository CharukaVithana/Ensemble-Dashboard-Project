"""Hybrid RF + ANN model (kept in its own file so joblib/Streamlit can load it)."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPRegressor


class HybridRFANN(BaseEstimator, RegressorMixin):
    """Random Forest feeds its prediction into an ANN (MLP) as an extra input.

    Out-of-fold (OOF) RF predictions are used for training the ANN, so the ANN
    does not see "too perfect" RF predictions (no leakage).
    """

    def __init__(self, cv=5, random_state=42):
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        rf = RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                   random_state=self.random_state)
        oof = cross_val_predict(clone(rf), X, y, cv=self.cv)
        self.mu_, self.sd_ = oof.mean(), oof.std() + 1e-9
        self.rf_ = rf.fit(X, y)
        self.ann_ = MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True,
                                 max_iter=600, random_state=self.random_state)
        self.ann_.fit(np.column_stack([X, (oof - self.mu_) / self.sd_]), y)
        return self

    def predict(self, X):
        rf_pred = (self.rf_.predict(X) - self.mu_) / self.sd_
        return self.ann_.predict(np.column_stack([X, rf_pred]))
