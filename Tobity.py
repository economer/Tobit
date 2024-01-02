import numpy as np
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.stats import norm

class Tobit(GenericLikelihoodModel):
    def __init__(self, endog, exog, left_censoring=-np.inf, right_censoring=np.inf):
        super(Tobit, self).__init__(endog, exog)
        self.left_censoring = left_censoring
        self.right_censoring = right_censoring

    def loglikeobs(self, params):
        beta = params[:-1]
        sigma = np.exp(params[-1])  # Ensure sigma is always positive
        xb = np.dot(self.exog, beta)
        y = self.endog

        # Indicator for censoring
        left_censored = y <= self.left_censoring
        right_censored = y >= self.right_censoring
        uncensored = ~left_censored & ~right_censored

        # Log likelihood for left-censored, right-censored, and uncensored
        ll_left = norm.logcdf((self.left_censoring - xb[left_censored]) / sigma) if self.left_censoring > -np.inf else 0
        ll_right = norm.logsf((self.right_censoring - xb[right_censored]) / sigma) if self.right_censoring < np.inf else 0
        ll_uncensored = norm.logpdf(y[uncensored], loc=xb[uncensored], scale=sigma)

        return np.sum(ll_left) + np.sum(ll_right) + np.sum(ll_uncensored)

    def fit(self, start_params=None, maxiter=16000, maxfun=5000,method = 'BFGS',**kwds):
        if start_params is None:
            # Consider using OLS to derive starting values for beta
            ols_res = sm.OLS(self.endog, self.exog).fit()
            start_params = np.append(ols_res.params, np.log(ols_res.scale))
        return super(Tobit, self).fit(start_params=start_params, method=method, maxiter=maxiter, maxfun=maxfun, **kwds)
