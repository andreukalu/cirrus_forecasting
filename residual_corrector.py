from sklearn.base import BaseEstimator, RegressorMixin, clone

class ResidualCorrector(BaseEstimator, RegressorMixin):
    def __init__(self, base_model = None, residual_model = None):
        self.base_model = base_model
        self.residual_model = residual_model

    def fit(self, X, y):
        self.base_model_ = clone(self.base_model)
        self.base_model_.fit(X,y)
        residuals = y - self.base_model_.predict(X)

        self.residual_model_ = clone(self.residual_model)
        self.residual_model_.fit(X,residuals)
        return self
    
    def predict(self,X):
        base_pred = self.base_model_.predict(X)
        residual_pred = self.residual_model_.predict(X)
        return base_pred + residual_pred