from typing import List, Dict, Optional

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .plots import plot_comparison_bars

__all__ = [
    'Linearizer',
    'CUPED',
    'PostStratification',
    'CUPAC',
]

def apply_linearization(num: np.ndarray, denom: np.ndarray, theta: float) -> np.ndarray:
    """Linearized metric for a single group.

    num:   ratio numerator per user (e.g. total session length)
    denom: ratio denominator per user (e.g. number of sessions)
    theta: global ratio estimate (usually taken from the control group)
    """
    return num - theta * denom

def Linearizer(test, control=None, use_pool_theta=False):
    """Hypothesis testing via ratio-metric linearization + t-test.

    test:           (num_array, denom_array) for the test group
    control:        (num_array, denom_array) for the control group
    use_pool_theta: if True, theta is estimated on the pool of both groups,
                    if False, theta is estimated only on the control group

    return: pvalue, delta (metric difference test - control)
    """
    num_t, denom_t = test

    if control is not None:
        num_c, denom_c = control
    else:
        num_c, denom_c = np.array([]), np.array([])

    if use_pool_theta:
        theta = (num_t.sum() + num_c.sum()) / (denom_t.sum() + denom_c.sum())
    else:
        theta = num_c.sum() / denom_c.sum()   # theta from the control group only

    lin_t = apply_linearization(num_t, denom_t, theta)
    if control is None:
        return lin_t

    lin_c = apply_linearization(num_c, denom_c, theta)

    return lin_t, lin_c, theta

def _ols_coef_inference(Z, y, coef, intercept, fit_intercept):
    """OLS inference for coefficients: standard errors, t- and p-values.

    Z:            design matrix WITHOUT the intercept column, shape (n, p)
    y:            target, shape (n,)
    coef:         coefficient estimates (without intercept), length p
    intercept:    intercept estimate
    fit_intercept: whether the intercept was accounted for during fitting

    The p-value is computed via a two-sided t-test for H0: coefficient = 0.
    For rank-deficient design matrices (full one-hot encoding + intercept)
    the pseudo-inverse is used, and the number of degrees of freedom is
    derived from the rank.

    return: (se, tvalues, pvalues) - arrays of length p (per coefficient).
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = Z.shape[0]
    if fit_intercept:
        Zd = np.column_stack([np.ones(n), Z])
        beta = np.concatenate([[intercept], np.asarray(coef, dtype=float)])
    else:
        Zd = Z
        beta = np.asarray(coef, dtype=float)
    rank = np.linalg.matrix_rank(Zd)
    dof = n - rank
    resid = y - Zd @ beta
    if dof <= 0:
        nan = np.full(len(coef), np.nan)
        return nan, nan, nan
    sigma2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.pinv(Zd.T @ Zd)
    se_all = np.sqrt(np.clip(np.diag(sigma2 * XtX_inv), 0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        t_all = beta / np.where(se_all > 0, se_all, np.nan)
    p_all = 2 * stats.t.sf(np.abs(t_all), df=dof)
    if fit_intercept:
        return se_all[1:], t_all[1:], p_all[1:]
    return se_all, t_all, p_all


class CUPED:
    """CUPED variance reduction via a single pre-experiment covariate.

    Fits a linear regression ``y ~ X`` on a single covariate and returns
    the residuals ``y - y_hat`` (the metric cleaned of the variance that is
    explained by the covariate).
    """

    def __init__(self):
        self.lreg = LinearRegression(copy_X=False, fit_intercept=True)

    def _check_X(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] > 1:
            raise ValueError('X should be one-dimensional')
        return X

    def fit(self, X, y):

        X = self._check_X(X)
        self.lreg.fit(X, y)
        self.theta = self.lreg.coef_[0]
        # quality measures (in-sample)
        y_arr = np.asarray(y)
        resid = y_arr - self.lreg.predict(X)
        var_y = y_arr.var()
        self.var_reduction_ = float(1 - resid.var() / var_y) if var_y > 0 else 0.0
        self.r2_ = float(self.lreg.score(X, y))
        # significance of theta
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        se, t, p = _ols_coef_inference(
            Xa, y_arr, self.lreg.coef_, self.lreg.intercept_, fit_intercept=True
        )
        self.se_ = float(se[0])
        self.tvalue_ = float(t[0])
        self.pvalue_ = float(p[0])
        return self

    def predict(self, X):
        X = self._check_X(X)
        return self.lreg.predict(X)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    def score(self, X, y, plot=True):
        """CUPED quality measures.

        Returns a dict:
        - ``var_reduction`` - share of removed metric variance ``1 - Var(resid)/Var(y)``.
          This is the main measure for CUPED: roughly the factor by which the
          variance of the effect estimate is reduced (in-sample it equals R-squared).
        - ``r2`` - coefficient of determination of the covariate-on-metric regression.
        - ``resid_std`` - standard deviation of the residuals (the metric after CUPED).

        Pass a hold-out sample to compute the measures out-of-sample.
        """
        y = np.asarray(y)
        X = self._check_X(X)
        resid = y - self.predict(X)
        var_y = y.var()
        var_r = resid.var()
        vr = float(1 - var_r / var_y) if var_y > 0 else 0.0
        if plot:
            plot_comparison_bars(
                baseline=var_y,
                values=[var_r],
                labels=['CUPED'],
                title='CUPED Performance',
                ylabel='Variance',
                figsize=(5, 3.5)
            )
        return {
            'var_reduction': vr,
            'r2': float(self.lreg.score(X, y)),
            'resid_std': float(resid.std()),
        }

    def cuped(self, X, y, fit=True):
        if fit:
            self.fit(X, y)
        return y - self.predict(X)


class PostStratification:
    """Post-stratification via linear regression on covariates.

    Works analogously to :class:`CUPED`: fits a linear regression
    ``y ~ X`` and returns the residuals ``y - y_hat`` (the metric cleaned
    of the variance explained by the covariates).

    String (categorical) columns are detected automatically and run through
    one-hot encoding (dummy variables). Numeric columns are passed into the
    regression as-is. If category values that were not seen during training
    arrive in ``predict``, they are simply ignored
    (``handle_unknown='ignore'`` - all dummy columns are set to zero for them).

    Parameters
    ----------
    cat_cols : list of str, optional
        Explicit list of categorical columns. If not provided and a
        :class:`pandas.DataFrame` is passed as input, categorical columns are
        detected automatically by dtype (object / category / string).
    fit_intercept : bool, default True
        Passed to :class:`sklearn.linear_model.LinearRegression`.
    """

    def __init__(self, cat_cols: Optional[List[str]] = None, fit_intercept: bool = True):
        self.cat_cols = cat_cols
        self.fit_intercept = fit_intercept
        self.pipeline: Optional[Pipeline] = None

    @staticmethod
    def _to_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.Series):
            return X.to_frame()
        elif isinstance(X, pd.DataFrame):
            return X
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])

    def _detect_cat_cols(self, X: pd.DataFrame) -> List[str]:
        if self.cat_cols is not None:
            return [c for c in self.cat_cols if c in X.columns]
        # auto-detect string/categorical columns by dtype
        return [
            c for c in X.columns
            if (pd.api.types.is_object_dtype(X[c])
                or isinstance(X[c].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(X[c]))
        ]

    def _build_pipeline(self, cat_cols: List[str], num_cols: List[str]) -> Pipeline:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        transformers = []
        if cat_cols:
            transformers.append(('cat', ohe, cat_cols))
        if num_cols:
            transformers.append(('num', 'passthrough', num_cols))
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
        )
        return Pipeline([
            ('prep', preprocessor),
            ('lreg', LinearRegression(copy_X=False, fit_intercept=self.fit_intercept)),
        ])

    def fit(self, X, y):
        X = self._to_frame(X)
        y = np.asarray(y)
        cat_cols = self._detect_cat_cols(X)
        num_cols = [c for c in X.columns if c not in cat_cols]
        self.cat_cols_ = cat_cols
        self.num_cols_ = num_cols
        self.pipeline = self._build_pipeline(cat_cols, num_cols)
        self.pipeline.fit(X, y)
        lreg = self.pipeline.named_steps['lreg']
        self.coef_ = lreg.coef_
        self.intercept_ = lreg.intercept_
        self.feature_names_ = list(
            self.pipeline.named_steps['prep'].get_feature_names_out()
        )
        # category frequencies on the training set (to find rare/useless ones)
        self._cat_counts = {
            c: X[c].value_counts(dropna=False).to_dict() for c in cat_cols
        }
        # quality measures (in-sample)
        resid = y - self.pipeline.predict(X)
        var_y = y.var()
        self.var_reduction_ = float(1 - resid.var() / var_y) if var_y > 0 else 0.0
        self.r2_ = float(self.pipeline.score(X, y))
        # coefficient significance (in-sample OLS inference)
        Z = self.pipeline.named_steps['prep'].transform(X)
        se, tvals, pvals = _ols_coef_inference(
            Z, y, lreg.coef_, lreg.intercept_, fit_intercept=self.fit_intercept
        )
        self.se_ = se
        self.tvalues_ = tvals
        self.pvalues_ = pvals
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError('PostStratification is not fitted yet, call fit() first')
        X = self._to_frame(X)
        return self.pipeline.predict(X)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    def post_stratification(self, X, y, fit=True):
        """Metric residuals after post-stratification: ``y - y_hat``."""
        if fit:
            self.fit(X, y)
        return np.asarray(y) - self.predict(X)

    def score(self, X, y):
        """Post-stratification quality measures.

        Returns a dict:
        - ``var_reduction`` - share of removed variance ``1 - Var(resid)/Var(y)``,
          the main usefulness measure (in-sample it equals R-squared).
        - ``r2`` - coefficient of determination of the regression.
        - ``resid_std`` - standard deviation of the residuals.

        Pass a hold-out sample for an honest out-of-sample evaluation.
        """
        y = np.asarray(y)
        resid = y - self.predict(X)
        var_y = y.var()
        vr = float(1 - resid.var() / var_y) if var_y > 0 else 0.0
        return {
            'var_reduction': vr,
            'r2': float(self.pipeline.score(self._to_frame(X), y)),
            'resid_std': float(resid.std()),
        }

    def coefficients(self) -> pd.DataFrame:
        """Table of regression coefficients across all features.

        For categorical columns the rows correspond to individual dummy
        categories (with their training frequency), for numeric columns - to
        the column itself. The ``se`` / ``tvalue`` / ``pvalue`` columns hold
        the OLS inference of coefficient significance (H0: coefficient = 0).
        """
        if self.pipeline is None:
            raise RuntimeError('PostStratification is not fitted yet, call fit() first')
        rows = []
        for name, coef, se, tval, pval in zip(
            self.feature_names_, self.coef_, self.se_, self.tvalues_, self.pvalues_
        ):
            if name.startswith('cat__'):
                rest = name[len('cat__'):]
                col, _, cat = rest.rpartition('_')
                count = self._cat_counts.get(col, {}).get(cat)
                if count is None:
                    # the category may be a number / contain '_' - try to match by value
                    for cand, cnt in self._cat_counts.get(col, {}).items():
                        if str(cand) == cat:
                            count = cnt
                            break
                rows.append({'column': col, 'category': cat, 'kind': 'cat',
                             'coef': coef, 'count': count,
                             'se': se, 'tvalue': tval, 'pvalue': pval})
            elif name.startswith('num__'):
                col = name[len('num__'):]
                rows.append({'column': col, 'category': None, 'kind': 'num',
                             'coef': coef, 'count': None,
                             'se': se, 'tvalue': tval, 'pvalue': pval})
            else:
                rows.append({'column': name, 'category': None, 'kind': '?',
                             'coef': coef, 'count': None,
                             'se': se, 'tvalue': tval, 'pvalue': pval})
        df = pd.DataFrame(rows)
        df['abs_coef'] = df['coef'].abs()
        return df.sort_values('abs_coef', ascending=False).reset_index(drop=True)

    def column_importance(self, X, y) -> pd.DataFrame:
        """Importance of each column via the leave-one-column-out method.

        For every column we re-fit the model without it and look at how much
        ``var_reduction`` drops. The larger the drop (``importance``), the more
        useful the column. Values near zero (or negative) are candidates for
        removal.
        """
        X = self._to_frame(X)
        y = np.asarray(y)
        full_vr = self.score(X, y)['var_reduction'] if self.pipeline is not None \
            else PostStratification(self.cat_cols, self.fit_intercept).fit(X, y).var_reduction_
        rows = []
        for col in X.columns:
            sub = PostStratification(
                cat_cols=([c for c in self.cat_cols if c != col]
                          if self.cat_cols is not None else None),
                fit_intercept=self.fit_intercept,
            )
            sub.fit(X.drop(columns=[col]), y)
            rows.append({'column': col,
                         'var_reduction_without': sub.var_reduction_,
                         'importance': full_vr - sub.var_reduction_})
        return (pd.DataFrame(rows)
                .sort_values('importance', ascending=False)
                .reset_index(drop=True))

    def find_useless_columns(self, X, y, threshold: float = 1e-4) -> List[str]:
        """Columns whose removal barely worsens var_reduction.

        Returns the list of columns with ``importance < threshold`` (see
        :meth:`column_importance`).
        """
        imp = self.column_importance(X, y)
        return imp.loc[imp['importance'] < threshold, 'column'].tolist()

    def find_useless_categories(self, min_count: int = 30,
                                coef_threshold: float = 1e-6,
                                alpha: float = 0.05) -> pd.DataFrame:
        """Useless / unreliable categories.

        Flags dummy categories that have:
        - a training frequency below ``min_count`` (the coefficient estimate
          is unreliable) - the ``rare`` flag;
        - a coefficient magnitude below ``coef_threshold`` (a negligible
          contribution to the prediction) - the ``negligible`` flag;
        - a coefficient p-value above ``alpha`` (statistically
          indistinguishable from zero) - the ``insignificant`` flag.

        Returns a subtable of :meth:`coefficients` with these flags for the
        categorical rows only.
        """
        df = self.coefficients()
        df = df[df['kind'] == 'cat'].copy()
        df['rare'] = df['count'].apply(
            lambda c: (c is None) or (c < min_count)
        )
        df['negligible'] = df['abs_coef'] < coef_threshold
        df['insignificant'] = df['pvalue'] > alpha
        mask = df['rare'] | df['negligible'] | df['insignificant']
        return df[mask].reset_index(drop=True)


class CUPAC:
    """CUPAC (Control Using Predictions As Covariates) on a linear model.

    Generalizes :class:`CUPED` to many (possibly noisy) pre-experiment
    covariates. Instead of using a single covariate directly, CUPAC first
    trains a (regularized) linear predictor of the metric ``y_hat = g(X)``
    and then applies a CUPED-style adjustment using that prediction as a
    single synthetic covariate::

        theta = Cov(y, y_hat) / Var(y_hat)
        y_adj = y - theta * (y_hat - mean(y_hat))

    Mean-centering keeps ``mean(y_adj) == mean(y)``. With ``theta`` chosen
    optimally the achievable variance reduction equals ``corr(y, y_hat) ** 2``.

    The interface mirrors :class:`CUPED` and :class:`PostStratification`:
    ``fit`` / ``predict`` / ``fit_predict`` / ``score`` plus the transform
    method :meth:`cupac`. As in the sibling classes, ``predict(X)`` returns
    the quantity that is subtracted from ``y``, so that
    ``cupac(X, y) == y - predict(X)``.

    String (categorical) columns are auto-detected and one-hot encoded;
    numeric columns are standardized (so that the regularization penalty
    treats features on a comparable scale).

    Parameters
    ----------
    regularization : {'none', 'ridge', 'lasso', 'elasticnet'}, default 'none'
        Which linear estimator backs the predictor ``g(X)``. Aliases
        ``'ols'`` / ``'linear'`` (none), ``'l2'`` (ridge), ``'l1'`` (lasso),
        ``'elastic'`` / ``'en'`` (elasticnet) are accepted.
    alpha : float, default 1.0
        Regularization strength for ridge / lasso / elasticnet. Ignored for
        the unregularized estimator.
    l1_ratio : float, default 0.5
        ElasticNet mixing parameter (0 = ridge-like, 1 = lasso-like). Only
        used when ``regularization='elasticnet'``.
    cat_cols : list of str, optional
        Explicit list of categorical columns. If not provided and a
        :class:`pandas.DataFrame` is passed, categorical columns are detected
        automatically by dtype (object / category / string).
    fit_intercept : bool, default True
        Passed to the underlying linear estimator.
    scale_numeric : bool, default True
        Standardize numeric columns before fitting. Recommended when a
        regularized estimator is used so that the penalty is scale-invariant.
    reg_kwargs : dict, optional
        Extra keyword arguments forwarded to the underlying sklearn estimator
        (e.g. ``max_iter`` for lasso / elasticnet).
    """

    def __init__(self,
                 regularization: str = 'none',
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 cat_cols: Optional[List[str]] = None,
                 fit_intercept: bool = True,
                 scale_numeric: bool = True,
                 reg_kwargs: Optional[Dict] = None):
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.cat_cols = cat_cols
        self.fit_intercept = fit_intercept
        self.scale_numeric = scale_numeric
        self.reg_kwargs = reg_kwargs or {}
        self.pipeline: Optional[Pipeline] = None

    @staticmethod
    def _to_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])

    def _detect_cat_cols(self, X: pd.DataFrame) -> List[str]:
        if self.cat_cols is not None:
            return [c for c in self.cat_cols if c in X.columns]
        # auto-detect string/categorical columns by dtype
        return [
            c for c in X.columns
            if (pd.api.types.is_object_dtype(X[c])
                or isinstance(X[c].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(X[c]))
        ]

    def _make_regressor(self):
        """Build the underlying linear estimator from the regularization choice."""
        reg = (self.regularization or 'none').lower()
        if reg in ('none', 'ols', 'linear'):
            return LinearRegression(copy_X=False, fit_intercept=self.fit_intercept,
                                    **self.reg_kwargs)
        if reg in ('ridge', 'l2'):
            return Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept,
                         **self.reg_kwargs)
        if reg in ('lasso', 'l1'):
            return Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept,
                         **self.reg_kwargs)
        if reg in ('elasticnet', 'elastic', 'en'):
            return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                              fit_intercept=self.fit_intercept, **self.reg_kwargs)
        raise ValueError(
            "regularization must be one of "
            "{'none', 'ridge', 'lasso', 'elasticnet'}, got "
            f"{self.regularization!r}"
        )

    def _build_pipeline(self, cat_cols: List[str], num_cols: List[str]) -> Pipeline:
        transformers = []
        if cat_cols:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            transformers.append(('cat', ohe, cat_cols))
        if num_cols:
            num_step = StandardScaler() if self.scale_numeric else 'passthrough'
            transformers.append(('num', num_step, num_cols))
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
        )
        return Pipeline([
            ('prep', preprocessor),
            ('reg', self._make_regressor()),
        ])

    def fit(self, X, y):
        X = self._to_frame(X)
        y = np.asarray(y, dtype=float)
        cat_cols = self._detect_cat_cols(X)
        num_cols = [c for c in X.columns if c not in cat_cols]
        self.cat_cols_ = cat_cols
        self.num_cols_ = num_cols
        self.pipeline = self._build_pipeline(cat_cols, num_cols)
        self.pipeline.fit(X, y)
        reg = self.pipeline.named_steps['reg']
        self.coef_ = np.asarray(reg.coef_, dtype=float).ravel()
        self.intercept_ = reg.intercept_
        self.feature_names_ = list(
            self.pipeline.named_steps['prep'].get_feature_names_out()
        )
        # base predictor output and the CUPED step on top of it
        p = np.asarray(self.pipeline.predict(X), dtype=float)
        self.prediction_mean_ = float(p.mean())
        var_p = p.var()
        self.theta_ = float(np.cov(y, p, ddof=0)[0, 1] / var_p) if var_p > 0 else 0.0
        # quality measures (in-sample)
        adj = y - self.theta_ * (p - self.prediction_mean_)
        var_y = y.var()
        self.var_reduction_ = float(1 - adj.var() / var_y) if var_y > 0 else 0.0
        self.predictor_r2_ = float(self.pipeline.score(X, y))
        # significance of theta from the CUPED regression y ~ y_hat
        intercept = float(y.mean() - self.theta_ * self.prediction_mean_)
        se, t, pv = _ols_coef_inference(
            p.reshape(-1, 1), y, [self.theta_], intercept, fit_intercept=True
        )
        self.se_ = float(se[0])
        self.tvalue_ = float(t[0])
        self.pvalue_ = float(pv[0])
        return self

    def predict_outcome(self, X):
        """Raw output of the base predictor ``g(X)`` (before the CUPED step)."""
        if self.pipeline is None:
            raise RuntimeError('CUPAC is not fitted yet, call fit() first')
        X = self._to_frame(X)
        return np.asarray(self.pipeline.predict(X), dtype=float)

    def predict(self, X):
        """The quantity subtracted from ``y``: ``theta * (g(X) - mean(g))``."""
        p = self.predict_outcome(X)
        return self.theta_ * (p - self.prediction_mean_)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    def cupac(self, X, y, fit=True):
        """Adjusted metric after CUPAC: ``y - theta * (g(X) - mean(g))``."""
        if fit:
            self.fit(X, y)
        return np.asarray(y, dtype=float) - self.predict(X)

    def score(self, X, y):
        """CUPAC quality measures.

        Returns a dict:
        - ``var_reduction`` - share of removed metric variance
          ``1 - Var(y_adj)/Var(y)``, the main measure (in-sample it equals
          ``corr(y, g(X)) ** 2``).
        - ``predictor_r2`` - coefficient of determination of the base
          predictor ``g(X)`` on the metric.
        - ``resid_std`` - standard deviation of the adjusted metric.
        - ``theta`` - the CUPED coefficient on the prediction.

        Pass a hold-out sample to compute the measures out-of-sample.
        """
        y = np.asarray(y, dtype=float)
        adj = y - self.predict(X)
        var_y = y.var()
        vr = float(1 - adj.var() / var_y) if var_y > 0 else 0.0
        return {
            'var_reduction': vr,
            'predictor_r2': float(self.pipeline.score(self._to_frame(X), y)),
            'resid_std': float(adj.std()),
            'theta': float(self.theta_),
        }

    def coefficients(self) -> pd.DataFrame:
        """Table of the base predictor coefficients across all features.

        For categorical columns the rows correspond to individual dummy
        categories, for numeric columns - to the column itself. These are the
        coefficients of the (possibly regularized) predictor ``g(X)``; with
        ridge / lasso / elasticnet they are shrunk and should be read as
        predictive weights rather than unbiased effect estimates.
        """
        if self.pipeline is None:
            raise RuntimeError('CUPAC is not fitted yet, call fit() first')
        rows = []
        for name, coef in zip(self.feature_names_, self.coef_):
            if name.startswith('cat__'):
                rest = name[len('cat__'):]
                col, _, cat = rest.rpartition('_')
                rows.append({'column': col, 'category': cat, 'kind': 'cat',
                             'coef': coef})
            elif name.startswith('num__'):
                col = name[len('num__'):]
                rows.append({'column': col, 'category': None, 'kind': 'num',
                             'coef': coef})
            else:
                rows.append({'column': name, 'category': None, 'kind': '?',
                             'coef': coef})
        df = pd.DataFrame(rows)
        df['abs_coef'] = df['coef'].abs()
        return df.sort_values('abs_coef', ascending=False).reset_index(drop=True)
