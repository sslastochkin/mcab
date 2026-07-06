import warnings

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .plots import plot_comparison_bars

__all__ = [
    'Linearizer',
    'CUPAC',
    'CUPED',
    'PostStratification',
    'coef_pvalue_table',
]


def apply_linearization(num: np.ndarray, denom: np.ndarray, theta: float) -> np.ndarray:
    """Linearized metric for a single group.

    num:   ratio numerator per user (e.g. total session length)
    denom: ratio denominator per user (e.g. number of sessions)
    theta: global ratio estimate (usually taken from the control group)
    """
    return num - theta * denom


def Linearizer(
    test: Tuple[np.ndarray, np.ndarray],
    control: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    use_pool_theta: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, float]]:
    """Ratio-metric linearization for hypothesis testing via a t-test.

    Turns a per-user ratio metric ``num / denom`` into a linearized
    per-user metric ``num - theta * denom`` that can be fed into an ordinary
    two-sample t-test while preserving the ratio's point estimate.

    Parameters
    ----------
    test : tuple of numpy.ndarray
        ``(num_array, denom_array)`` for the test group: the ratio numerator
        and denominator per user (e.g. total session length and number of
        sessions).
    control : tuple of numpy.ndarray, optional
        ``(num_array, denom_array)`` for the control group. If ``None``, only
        the test group is linearized and returned.
    use_pool_theta : bool, default False
        If ``True``, ``theta`` is estimated on the pooled data of both groups;
        if ``False``, ``theta`` is estimated only on the control group.

    Returns
    -------
    numpy.ndarray or tuple
        If ``control`` is ``None``, the linearized test-group array. Otherwise
        a tuple ``(lin_test, lin_control, theta)`` with the linearized arrays of
        both groups and the ``theta`` used.
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

# ------------------------------------------------------------------ inference

def _hc_weights(
    resid: np.ndarray,
    Zd: np.ndarray,
    XtX_inv: np.ndarray,
    cov_type: str,
    n: int,
    rank: int,
) -> np.ndarray:
    """Per-observation squared-residual weights for an HC sandwich estimator.

    Implements the White / MacKinnon-White family used by statsmodels:
    - 'HC0': e_i^2
    - 'HC1': n / (n - rank) * e_i^2
    - 'HC2': e_i^2 / (1 - h_i)
    - 'HC3': e_i^2 / (1 - h_i)^2
    where ``h_i`` are the leverages (diagonal of the hat matrix).
    """
    e2 = resid ** 2
    ct = cov_type.upper()
    if ct == 'HC0':
        return e2
    if ct == 'HC1':
        return (n / (n - rank)) * e2
    # HC2 / HC3 need leverages
    hat = np.einsum('ij,jk,ik->i', Zd, XtX_inv, Zd)
    hat = np.clip(hat, 0.0, 1.0 - 1e-10)
    if ct == 'HC2':
        return e2 / (1.0 - hat)
    if ct == 'HC3':
        return e2 / (1.0 - hat) ** 2
    raise ValueError(f"unknown cov_type {cov_type!r}; expected one of "
                     f"'HC0','HC1','HC2','HC3','nonrobust'")


def _sandwich_cov(
    Zd: np.ndarray,
    resid: np.ndarray,
    XtX_inv: np.ndarray,
    cov_type: str,
    dof: float,
    rank: int,
) -> np.ndarray:
    """Coefficient covariance for a linear model given ``(Z'Z)^-1`` (via pinv).

    ``cov_type='nonrobust'`` returns the classical homoskedastic covariance
    ``sigma^2 (Z'Z)^-1``; any 'HC*' value returns the heteroskedasticity-robust
    sandwich ``(Z'Z)^-1 Z' diag(omega) Z (Z'Z)^-1``.
    """
    n = Zd.shape[0]
    if str(cov_type).lower() == 'nonrobust':
        sigma2 = float(resid @ resid) / dof
        return sigma2 * XtX_inv
    w = _hc_weights(resid, Zd, XtX_inv, cov_type, n, rank)
    meat = Zd.T @ (Zd * w[:, None])
    return XtX_inv @ meat @ XtX_inv


def _ols_coef_inference(
    Z: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    fit_intercept: bool,
    cov_type: str = 'HC3',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS inference for coefficients: robust standard errors, t- and p-values.

    Z:             design matrix WITHOUT the intercept column, shape (n, p)
    y:             target, shape (n,)
    coef:          coefficient estimates (without intercept), length p
    intercept:     intercept estimate
    fit_intercept: whether the intercept was accounted for during fitting
    cov_type:      covariance estimator:
                   - 'HC0'/'HC1'/'HC2'/'HC3': heteroskedasticity-robust
                     (White / MacKinnon-White) sandwich standard errors;
                   - 'nonrobust': classical homoskedastic OLS covariance.
                   Default 'HC3' is the recommended small-sample robust choice.

    Standard errors are computed with the sandwich estimator
    ``(Z'Z)^-1 Z' diag(omega) Z (Z'Z)^-1`` where the weights ``omega`` depend on
    ``cov_type`` (HC0: e^2, HC1: n/(n-k) e^2, HC2: e^2/(1-h), HC3: e^2/(1-h)^2).
    These match statsmodels' ``OLS(...).fit(cov_type=...)`` on full-rank designs
    but, unlike statsmodels, the pseudo-inverse keeps the estimator well-defined
    for the rank-deficient design matrices (full one-hot + intercept) used here.
    The p-value is a two-sided t-test for H0: coefficient = 0.

    return: (se, tvalues, pvalues) - arrays of length p.
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = Z.shape[0]
    coef = np.asarray(coef, dtype=float).ravel()
    if fit_intercept:
        Zd = np.column_stack([np.ones(n), Z])
        beta = np.concatenate([[float(intercept)], coef])
    else:
        Zd, beta = Z, coef
    rank = np.linalg.matrix_rank(Zd)
    dof = n - rank
    resid = y - Zd @ beta
    if dof <= 0:
        nan = np.full(len(coef), np.nan)
        return nan, nan, nan
    XtX_inv = np.linalg.pinv(Zd.T @ Zd)
    cov = _sandwich_cov(Zd, resid, XtX_inv, cov_type=cov_type,
                        dof=dof, rank=rank)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        t = beta / np.where(se > 0, se, np.nan)
    p = 2 * stats.t.sf(np.abs(t), df=dof)
    return se, t, p


def _ridge_coef_inference(
    Z: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    alpha: float,
    fit_intercept: bool,
    cov_type: str = 'HC3',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust inference for ridge coefficients (sandwich formula).

    The ridge estimator is linear in ``y``: with ``A = (Z'Z + Lambda)^-1``
    (the intercept left unpenalized) ``beta_hat = A Z' y``. Its covariance is
    the sandwich ``A Z' diag(omega) Z A``. With ``cov_type='nonrobust'`` the
    homoskedastic weights ``omega_i = sigma^2`` reproduce the classical
    ``sigma^2 A (Z'Z) A``; with an 'HC*' value the weights are the
    heteroskedasticity-robust squared residuals (optionally leverage-adjusted
    using the ridge hat matrix ``H = Z A Z'``), giving standard errors that are
    valid under heteroskedasticity.

    Note: ridge estimates are biased; these SEs describe the variability of
    the shrunk estimator, not of an unbiased effect. The effective dof are
    ``n - tr(H)``.
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = Z.shape[0]
    coef = np.asarray(coef, dtype=float).ravel()
    if fit_intercept:
        Zd = np.column_stack([np.ones(n), Z])
        beta = np.concatenate([[float(intercept)], coef])
        penalty = np.concatenate([[0.0], np.full(len(coef), float(alpha))])
    else:
        Zd, beta = Z, coef
        penalty = np.full(len(coef), float(alpha))
    ZtZ = Zd.T @ Zd
    A = np.linalg.pinv(ZtZ + np.diag(penalty))
    # ridge hat matrix H = Z A Z'; effective dof = n - tr(H)
    hat = np.einsum('ij,jk,ik->i', Zd, A, Zd)
    dof = n - float(hat.sum())
    resid = y - Zd @ beta
    if dof <= 0:
        nan = np.full(len(coef), np.nan)
        return nan, nan, nan
    if str(cov_type).lower() == 'nonrobust':
        sigma2 = float(resid @ resid) / dof
        cov = sigma2 * (A @ ZtZ @ A)
    else:
        ct = cov_type.upper()
        e2 = resid ** 2
        if ct == 'HC0':
            w = e2
        elif ct == 'HC1':
            w = (n / dof) * e2
        elif ct == 'HC2':
            w = e2 / np.clip(1.0 - hat, 1e-10, None)
        elif ct == 'HC3':
            w = e2 / np.clip(1.0 - hat, 1e-10, None) ** 2
        else:
            raise ValueError(f"unknown cov_type {cov_type!r}; expected one of "
                             f"'HC0','HC1','HC2','HC3','nonrobust'")
        meat = Zd.T @ (Zd * w[:, None])
        cov = A @ meat @ A
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        t = beta / np.where(se > 0, se, np.nan)
    p = 2 * stats.t.sf(np.abs(t), df=dof)
    return se, t, p


def _logistic_coef_inference(
    Z: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    fit_intercept: bool,
    cov_type: str = 'HC0',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust Wald inference for logistic-regression coefficients.

    ``cov_type='nonrobust'`` uses the model-based inverse Fisher information
    ``(Z' diag(p(1-p)) Z)^-1``. Any 'HC*' value returns the robust
    ("sandwich" / Huber-White) covariance
    ``B^-1 (Z' diag(e_i^2) Z) B^-1`` with ``B = Z' diag(p(1-p)) Z`` and
    residuals ``e_i = y_i - p_i``, which is consistent even if the Bernoulli
    variance is misspecified (quasi-MLE). Significance is a two-sided Wald
    z-test. Matches statsmodels' ``GLM(..., Binomial()).fit(cov_type='HC0')``.
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = Z.shape[0]
    coef = np.asarray(coef, dtype=float).ravel()
    if fit_intercept:
        Zd = np.column_stack([np.ones(n), Z])
        beta = np.concatenate([[float(intercept)], coef])
    else:
        Zd, beta = Z, coef
    eta = Zd @ beta
    p = np.where(eta >= 0, 1.0 / (1.0 + np.exp(-eta)),
                 np.exp(eta) / (1.0 + np.exp(eta)))
    w = np.clip(p * (1.0 - p), 1e-12, None)
    bread = np.linalg.pinv(Zd.T @ (Zd * w[:, None]))
    if str(cov_type).lower() == 'nonrobust':
        cov = bread
    else:
        if cov_type.upper() not in ('HC0', 'HC1', 'HC2', 'HC3'):
            raise ValueError(f"unknown cov_type {cov_type!r}; expected one of "
                             f"'HC0','HC1','HC2','HC3','nonrobust'")
        # robust "meat": squared score contributions e_i^2 * z_i z_i'
        resid = y - p
        meat = Zd.T @ (Zd * (resid ** 2)[:, None])
        cov = bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        z = beta / np.where(se > 0, se, np.nan)
    pv = 2 * stats.norm.sf(np.abs(z))
    return se, z, pv


# ------------------------------------------------------------ coef p-values

def _final_estimator(model: Any) -> Any:
    """Return the terminal estimator of a (possibly Pipeline-wrapped) model.

    Handles plain sklearn :class:`~sklearn.pipeline.Pipeline` objects as well
    as :class:`CUPAC` instances (and its subclasses), which store the fitted
    pipeline in their ``.pipe`` attribute.
    """
    # CUPAC / CUPED / PostStratification — delegate to the inner pipeline
    if hasattr(model, 'pipe') and isinstance(getattr(model, 'pipe', None), Pipeline):
        return model.pipe[-1]
    return model[-1] if isinstance(model, Pipeline) else model


def _design_matrix(model: Any, X: Any) -> np.ndarray:
    """Design matrix seen by the terminal estimator.

    If ``model`` is a Pipeline (or a CUPAC instance whose ``.pipe`` is a
    Pipeline), every preprocessing step but the last is applied to ``X``;
    otherwise ``X`` is returned as a 2-D float array.
    """
    # Resolve the actual sklearn Pipeline
    pipe = None
    if isinstance(model, Pipeline):
        pipe = model
    elif hasattr(model, 'pipe') and isinstance(getattr(model, 'pipe', None), Pipeline):
        pipe = model.pipe
        # Normalise X the same way the model's fit() does:
        # CUPED uses _check_X (Series → ndarray → reshape → col 'x0'),
        # other CUPAC subclasses use _to_frame.
        if hasattr(model, '_check_X'):
            X = model._to_frame(model._check_X(X))
        elif hasattr(model, '_to_frame'):
            X = model._to_frame(X)

    if pipe is not None and len(pipe.steps) > 1:
        Xt = X
        for _, step in pipe.steps[:-1]:
            Xt = step.transform(Xt)
        # OHE with sparse_output=True returns a csr_matrix — convert it
        try:
            Z = np.asarray(Xt, dtype=float)
        except (TypeError, ValueError):
            Z = np.asarray(Xt.toarray(), dtype=float)
    else:
        Z = np.asarray(X, dtype=float)
    return Z.reshape(-1, 1) if Z.ndim == 1 else Z

def _strip_known_prefixes(names: List[str], prep: Pipeline) -> List[str]:
    """Strip sklearn's step/transformer name prefixes from feature names.

    ``sklearn`` prefixes the names returned by ``get_feature_names_out`` with
    the name of the pipeline step and, inside a
    :class:`~sklearn.compose.ColumnTransformer`, the name of the inner
    transformer (e.g. ``'prep__cat__color'``). This removes the longest
    matching known prefix from each name so the result is human-readable.

    Parameters
    ----------
    names : list of str
        Raw feature names produced by the preprocessing pipeline.
    prep : sklearn.pipeline.Pipeline
        The preprocessing pipeline (``model[:-1]``, i.e. every step except the
        final regressor) whose step and transformer names define the prefixes.

    Returns
    -------
    list of str
        Feature names with a single known prefix removed where present.
    """
    # prep = model[:-1] — Pipeline без последнего шага (регрессора)
    prefixes = []
    for step_name, step in prep.steps:
        prefixes.append(step_name + '__')
        # ColumnTransformer: собрать имена внутренних трансформеров
        sub = getattr(step, 'transformers_', None) or getattr(step, 'transformers', [])
        for t_name, _, _ in sub:
            prefixes.append(t_name + '__')
    # длинные префиксы приоритетнее — сортируем
    prefixes.sort(key=len, reverse=True)
    result = []
    for name in names:
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        result.append(name)
    return result

def _feature_names(model: Any, Z: np.ndarray, X: Any = None) -> List[str]:
    """Best-effort feature names for the design-matrix columns.

    Works for plain sklearn :class:`~sklearn.pipeline.Pipeline` objects and
    for :class:`CUPAC` instances (and subclasses) that store their fitted
    pipeline in ``.pipe``.

    When no pipeline preprocessing is present, tries to use column names from
    the original ``X`` (Series name or DataFrame columns).
    """
    # Resolve the sklearn Pipeline (plain or wrapped inside CUPAC)
    pipe = None
    if isinstance(model, Pipeline):
        pipe = model
    elif hasattr(model, 'pipe') and isinstance(getattr(model, 'pipe', None), Pipeline):
        pipe = model.pipe

    if pipe is not None and len(pipe.steps) > 1:
        prep = pipe[:-1]
        if hasattr(prep, 'get_feature_names_out'):
            try:
                raw = list(prep.get_feature_names_out())
                names = _strip_known_prefixes(raw, prep)
                if len(names) == Z.shape[1]:
                    # If the pipeline produced generic names like x0, x1, ...
                    # and the caller supplied original column names via X,
                    # prefer the original names (e.g. Series.name for CUPED).
                    if X is not None and all(n == f'x{i}' for i, n in enumerate(names)):
                        if isinstance(X, pd.Series) and X.name is not None and len(names) == 1:
                            names = [str(X.name)]
                        elif isinstance(X, pd.DataFrame) and list(X.columns) != names:
                            # Only override when DataFrame columns are not also x0, x1...
                            orig = list(X.columns)
                            if len(orig) == len(names):
                                names = orig
                    return names
            except Exception:
                pass

    # Fallback: try to use column names from the original X
    if X is not None:
        if isinstance(X, pd.DataFrame) and len(X.columns) == Z.shape[1]:
            return list(X.columns)
        if isinstance(X, pd.Series) and X.name is not None and Z.shape[1] == 1:
            return [str(X.name)]

    return [f'x{i}' for i in range(Z.shape[1])]


def coef_pvalue_table(
    model: Any,
    X: Any,
    y: Any,
    cov_type: Optional[str] = None,
    feature_names: Optional[List[str]] = None,
    sort_by: str = 'significance',
) -> pd.DataFrame:
    """Coefficient table with robust standard errors and p-values.

    Dispatches on the fitted estimator type and computes robust (sandwich)
    inference for its coefficients:

    - :class:`~sklearn.linear_model.LinearRegression` -> :func:`_ols_coef_inference`
      (heteroskedasticity-robust HC standard errors);
    - :class:`~sklearn.linear_model.Ridge` -> :func:`_ridge_coef_inference`
      (robust ridge sandwich standard errors);
    - :class:`~sklearn.linear_model.LogisticRegression` -> :func:`_logistic_coef_inference`
      (robust Huber-White Wald standard errors).

    All three underlying estimators return *robust* standard errors by default,
    matching statsmodels' ``cov_type='HC*'`` on full-rank designs while still
    supporting the rank-deficient (full one-hot + intercept) design matrices via
    the pseudo-inverse. ``linearmodels`` / ``statsmodels`` are intentionally not
    used here because neither exposes a robust covariance for the *penalized*
    ridge estimator, and statsmodels breaks on the rank-deficient designs this
    module relies on.

    Parameters
    ----------
    model : fitted estimator or Pipeline
        A fitted ``LinearRegression`` / ``Ridge`` / ``LogisticRegression`` (or a
        :class:`~sklearn.pipeline.Pipeline` ending in one of them).
    X : array-like or DataFrame
        Features. When ``model`` is a Pipeline, the preprocessing steps are
        applied to obtain the design matrix; otherwise ``X`` is used as is.
    y : array-like
        Target used to fit the model (needed to evaluate residuals).
    cov_type : str, optional
        Robust covariance estimator forwarded to the inference function
        ('HC0'/'HC1'/'HC2'/'HC3', or 'nonrobust'). Defaults to 'HC3' for
        OLS/Ridge and 'HC0' for logistic regression.
    feature_names : list of str, optional
        Names for the design-matrix columns; auto-derived from the pipeline (or
        generated as ``x0, x1, ...``) when omitted.
    sort_by : {'significance', 'effect'}, default 'significance'
        Criterion used to rank feature rows (the ``(Intercept)`` row, when
        present, is always pinned to the top regardless of this setting):

        ``'significance'``
            Sort by ``|stat|`` descending (primary), then by ``|coef|``
            descending (tiebreaker).  Equivalent to sorting by ascending
            p-value but **robust to p-value underflow**: when many features
            have ``pvalue == 0.0`` (float64 underflow for ``|stat| > ~37``),
            the test statistic still distinguishes them.

        ``'effect'``
            Sort by the *standardised* coefficient ``|coef| * std(feature)``
            descending (primary), then by ``|stat|`` descending (tiebreaker).
            Useful in large experiments where almost everything is significant
            and practical importance matters more than statistical significance.

    Returns
    -------
    pandas.DataFrame
        Columns ``feature / coef / se / stat / pvalue`` (``stat`` is a t-value
        for OLS/Ridge and a z-value for logistic regression).  The
        ``(Intercept)`` row, when present, is always the first row.

    Raises
    ------
    TypeError
        If the terminal estimator is not one of the three supported types.
    ValueError
        If ``sort_by`` is not one of the supported values.
    """
    est = _final_estimator(model)
    y = np.asarray(y, dtype=float)
    Z = _design_matrix(model, X)

    coef = np.ravel(getattr(est, 'coef_', np.array([]))).astype(float)
    intercept = np.ravel(getattr(est, 'intercept_', 0.0)).astype(float)
    intercept = float(intercept[0]) if intercept.size else 0.0
    fit_intercept = bool(getattr(est, 'fit_intercept', True))

    if isinstance(est, LogisticRegression):
        stat_name, kind = 'z', 'logistic'
        ct = cov_type or 'HC0'
        se, stat, pv = _logistic_coef_inference(
            Z, y, coef, intercept, fit_intercept, cov_type=ct)
    elif isinstance(est, Ridge):
        stat_name, kind = 't', 'ridge'
        ct = cov_type or 'HC3'
        alpha = float(getattr(est, 'alpha', 1.0))
        se, stat, pv = _ridge_coef_inference(
            Z, y, coef, intercept, alpha, fit_intercept, cov_type=ct)
    elif isinstance(est, LinearRegression):
        stat_name, kind = 't', 'ols'
        ct = cov_type or 'HC3'
        se, stat, pv = _ols_coef_inference(
            Z, y, coef, intercept, fit_intercept, cov_type=ct)
    else:
        raise TypeError(
            f"coef_pvalue_table supports LinearRegression, Ridge and "
            f"LogisticRegression (optionally inside a Pipeline), got "
            f"{type(est).__name__}"
        )

    # se/stat/pv now include intercept at index 0 when fit_intercept=True
    if fit_intercept:
        se_coef, stat_coef, pv_coef = se[1:], stat[1:], pv[1:]
        se_int, stat_int, pv_int = float(se[0]), float(stat[0]), float(pv[0])
    else:
        se_coef, stat_coef, pv_coef = se, stat, pv

    names = feature_names if feature_names is not None else _feature_names(model, Z, X)
    if len(names) != len(coef):
        names = [f'x{i}' for i in range(len(coef))]

    df = pd.DataFrame({
        'feature': names,
        'coef': coef,
        'se': np.asarray(se_coef, dtype=float),
        'stat': np.asarray(stat_coef, dtype=float),
        'pvalue': np.asarray(pv_coef, dtype=float),
    })

    if fit_intercept:
        intercept_row = pd.DataFrame([{
            'feature': '(Intercept)',
            'coef': intercept,
            'se': se_int,
            'stat': stat_int,
            'pvalue': pv_int,
        }])
        df = pd.concat([intercept_row, df], ignore_index=True)

    _SORT_BY_VALUES = {'significance', 'effect'}
    if sort_by not in _SORT_BY_VALUES:
        raise ValueError(
            f"sort_by must be one of {sorted(_SORT_BY_VALUES)!r}, got {sort_by!r}"
        )

    # Pin the intercept row first; sort only feature rows.
    if fit_intercept:
        df_intercept = df[df['feature'] == '(Intercept)'].copy()
        df_features  = df[df['feature'] != '(Intercept)'].copy()
    else:
        df_intercept = pd.DataFrame(columns=df.columns)
        df_features  = df.copy()

    df_features['_abs_stat'] = df_features['stat'].abs()
    df_features['_abs_coef'] = df_features['coef'].abs()

    if sort_by == 'significance':
        # Primary: |stat| ↓ (robust to p-value float64 underflow);
        # tiebreaker: |coef| ↓
        df_features = df_features.sort_values(
            ['_abs_stat', '_abs_coef'], ascending=[False, False]
        )
    else:  # 'effect'
        # Primary: standardised coefficient |coef|·std(feature) ↓;
        # tiebreaker: |stat| ↓ (significance as secondary signal).
        # Z columns are aligned 1-to-1 with coef (no intercept column in Z).
        feature_stds = np.std(Z, axis=0)            # shape (n_features,)
        std_effect   = np.abs(coef) * feature_stds  # shape (n_features,)
        df_features['_std_effect'] = std_effect
        df_features = df_features.sort_values(
            ['_std_effect', '_abs_stat'], ascending=[False, False]
        )
        df_features = df_features.drop(columns='_std_effect')

    df_features = df_features.drop(columns=['_abs_stat', '_abs_coef'])

    if df_intercept.empty:
        return df_features.reset_index(drop=True)
    return pd.concat([df_intercept, df_features], ignore_index=True)

class CUPAC:
    """CUPAC variance reduction via a fitted predictor of the target.

    CUPAC (Control Using Predictions As Covariates) builds a preprocessing +
    regression pipeline that predicts the experiment metric from pre-experiment
    covariates and returns the residuals, which have lower variance than the
    raw metric while keeping the treatment effect unbiased.

    The estimator wraps an sklearn regressor (or classifier) with an optional
    :class:`~sklearn.compose.ColumnTransformer` that one-hot encodes categorical
    columns and scales numeric ones. Preprocessing options are supplied at
    :meth:`fit` time.

    Parameters
    ----------
    model : str or estimator
        Either a key of :attr:`_MODELS` (``'ols'``, ``'log'``, ``'ridge'``,
        ``'lasso'``, ``'elastic_net'``) selecting a default sklearn estimator,
        or an already-constructed estimator instance.
    name : str, optional
        Name of the regression step in the pipeline. Defaults to the model key
        when ``model`` is a known string, otherwise ``'model'``.

    Attributes
    ----------
    _MODELS : dict
        Mapping of string keys to sklearn estimator classes.
    pipe : sklearn.pipeline.Pipeline or None
        The fitted pipeline; ``None`` until :meth:`fit` is called.
    """

    _MODELS = {
        'ols':         LinearRegression,
        'log':         LogisticRegression,
        'ridge':       Ridge,
        'lasso':       Lasso,
        'elastic_net': ElasticNet
    }

    def __init__(
        self,
        model: Union[str, Any],
        name: Optional[str] = None,
    ) -> None:
        if name is None:
            if model in self._MODELS:
                name = model
            else:
                name = 'model'
        self.name = name

        if isinstance(model, str):
            model = self._MODELS[model]()

        self._model =  model
        
        self.pipe = None
        self.cat_cols = None
        self.cat_encoder = None
        self.excluded_categories = None
        self.scaler = None
        self.no_scale_cols = None
        self.resids = None

    @staticmethod
    def _to_frame(X: Any) -> pd.DataFrame:
        """Coerce array-like / Series / DataFrame input into a DataFrame.

        1-D inputs are reshaped into a single column; anonymous columns are
        named ``x0, x1, ...``.
        """
        if isinstance(X, pd.Series):
            return X.to_frame()
        if isinstance(X, pd.DataFrame):
            return X
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
    
    def _detect_cat_cols(self, X: pd.DataFrame) -> List[str]:
        """Resolve which columns of ``X`` are treated as categorical.

        If :attr:`cat_cols` is ``True``, object / categorical / string columns
        are auto-detected; if it is an explicit list, only those present in
        ``X`` are kept; otherwise no columns are categorical.
        """
        if self.cat_cols == True:
            cat_cols = [c for c in X.columns
                    if (pd.api.types.is_object_dtype(X[c])
                        or isinstance(X[c].dtype, pd.CategoricalDtype)
                        or pd.api.types.is_string_dtype(X[c]))]
        elif self.cat_cols:
            cat_cols = [c for c in self.cat_cols if c in X.columns]
        else:
            cat_cols = []
        return cat_cols
            
    def _get_categories(
        self, cat_cols: List[str], X: pd.DataFrame
    ) -> List[np.ndarray]:
        """Explicit category levels per categorical column for the encoder.

        For each column in ``cat_cols`` the observed non-null unique values are
        collected, dropping any levels listed in :attr:`excluded_categories`.
        """
        if not cat_cols:
            return []
        return [
            np.array([v for v in pd.unique(X[c].dropna())
                        if v not in set(self.excluded_categories.get(c, []))],
                        dtype=object)
            for c in cat_cols
        ]
    
    def _get_num_cols(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Split numeric columns into scaled vs. passthrough groups.

        Returns ``(num_cols_scale, num_cols_pass)``. When no scaler is set,
        every numeric column is passthrough. Otherwise columns listed in
        :attr:`no_scale_cols` are passed through and the rest are scaled.
        """
        num_cols_all = X.select_dtypes(include='number').columns.tolist()
        if self.scaler is None:
            return [], num_cols_all

        num_cols_scale = [c for c in num_cols_all if c not in self.no_scale_cols]
        num_cols_pass  = [c for c in num_cols_all if c in self.no_scale_cols]
        return num_cols_scale, num_cols_pass

    def build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Assemble the preprocessing + regression pipeline for ``X``.

        Builds a :class:`~sklearn.compose.ColumnTransformer` with (optional)
        categorical encoding, numeric scaling and passthrough steps according
        to the options set on the instance, followed by the regression step.

        Parameters
        ----------
        X : pandas.DataFrame
            Training features used to infer categorical/numeric columns.

        Returns
        -------
        sklearn.pipeline.Pipeline
            The unfitted pipeline ending in the regression estimator.
        """
        pipe_list = []
        transformers = []
        
        cat_cols   = self._detect_cat_cols(X)
        categories = self._get_categories(cat_cols, X)

        if cat_cols and self.cat_encoder is not None:
            transformers.append(('cat', self.cat_encoder(categories=categories), cat_cols))

        num_cols_scale, num_cols_pass = self._get_num_cols(X)
        if num_cols_scale:
            transformers.append(('scale', self.scaler, num_cols_scale))

        if num_cols_pass:
            transformers.append(('pass', 'passthrough', num_cols_pass))

        if transformers:
            covered_cols = set(cat_cols) | set(num_cols_scale) | set(num_cols_pass)
            dropped_cols = [c for c in X.columns if c not in covered_cols]
            if dropped_cols:
                warnings.warn(
                    f"{self.__class__.__name__}: the following columns are not "
                    f"assigned to any transformer and will be silently dropped "
                    f"(remainder='drop'): {dropped_cols}. "
                    f"Categorical columns require cat_cols=True or "
                    f"cat_cols=[...] with cat_encoder to be included.",
                    UserWarning,
                    stacklevel=3,
                )
            prep = ColumnTransformer(transformers=transformers, remainder='drop')
            pipe_list.append(('prep', prep))

        pipe_list.append((self.name, self._model))
        return Pipeline(pipe_list)

    def __repr__(self) -> str:
        fitted = 'fitted' if self.pipe is not None else 'unfitted'
        return f'{self.__class__.__name__}(name="{self.name}", {fitted})'

    def fit(
        self,
        X: Any,
        y: Any,
        cat_cols: Optional[Union[bool, List[str]]] = None,
        cat_encoder: Optional[Callable] = None,
        excluded_categories: Optional[Dict[str, List]] = None,
        scaler: Optional[Any] = None,
        no_scale_cols: Optional[List[str]] = None,
        fit_kwargs: Optional[Dict] = None
    ) -> 'CUPAC':
        """Fit the CUPAC pipeline on features ``X`` and target ``y``.

        Parameters
        ----------
        X : array-like, Series or DataFrame
            Pre-experiment covariates used to predict the metric.
        y : array-like
            Target metric to be adjusted.
        cat_cols : bool or list of str, optional
            Categorical columns to encode: ``True`` to auto-detect, or an
            explicit list of column names.  When provided without
            ``cat_encoder``, a :class:`~sklearn.preprocessing.OneHotEncoder`
            is created automatically (``drop='first'`` for
            :class:`~sklearn.linear_model.LinearRegression` with
            ``fit_intercept=True``, ``drop=None`` otherwise).
        cat_encoder : callable, optional
            Factory returning an encoder that accepts a ``categories`` keyword
            argument (e.g. ``lambda cats: OneHotEncoder(categories=cats)``).
            Overrides the default auto-encoder when provided alongside
            ``cat_cols``.
        excluded_categories : dict of {str: list}, optional
            Mapping ``column -> levels`` of category levels to drop from the
            encoder's known categories before fitting.
        scaler : estimator instance, optional
            Scaler applied to numeric columns (e.g. ``StandardScaler()``).
            When ``None`` numeric columns are passed through unscaled.
        no_scale_cols : list of str, optional
            Numeric columns to pass through without scaling even when a
            ``scaler`` is provided.
        fit_kwargs : dict, optional
            Extra keyword arguments forwarded to the pipeline's ``fit`` call
            (e.g. sample weights via ``{'<step>__sample_weight': w}``).

        Returns
        -------
        CUPAC
            The fitted instance (``self``).

        Raises
        ------
        ValueError
            If ``cat_encoder`` is supplied without ``cat_cols``.
        """
        if cat_cols:
            if cat_encoder is None:
                # Auto-create a OneHotEncoder mirroring PostStratification.
                drop = None
                if getattr(self._model, 'fit_intercept', False):
                    drop = 'first'
                cat_encoder = lambda categories: OneHotEncoder(  # noqa: E731
                    categories=categories,
                    drop=drop,
                    sparse_output=False,
                    handle_unknown='ignore'
                )

        if cat_encoder is not None:
            if cat_cols is None:
                raise ValueError('cat_encoder is not None but cat_cols is None')
            
        self.cat_cols = cat_cols or []
        self.cat_encoder = cat_encoder
        self.excluded_categories = excluded_categories or {}
        self.scaler = scaler
        self.no_scale_cols = no_scale_cols or []

        X = self._to_frame(X)

        self.pipe = self.build_pipeline(X)
        fit_kwargs = fit_kwargs or {}
        self.pipe.fit(X, y, **fit_kwargs)

        return self
    
    def predict(self, X: Any) -> np.ndarray:
        """Predict the target for ``X`` using the fitted pipeline.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.pipe is None:
            raise ValueError('Model is not fitted')
        X = self._to_frame(X)
        return self.pipe.predict(X)

    def fit_predict(
        self,
        X: Any,
        y: Any,
        cat_cols: Optional[Union[bool, List[str]]] = None,
        cat_encoder: Optional[Callable] = None,
        excluded_categories: Optional[Dict[str, List]] = None,
        scaler: Optional[Any] = None,
        no_scale_cols: Optional[List[str]] = None,
        fit_kwargs: Optional[Dict] = None
    ) -> np.ndarray:
        """Fit on ``(X, y)`` and return in-sample predictions for the same ``X``.

        Equivalent to calling :meth:`fit` followed by :meth:`predict`.  All
        parameters are forwarded to :meth:`fit` unchanged.

        Parameters
        ----------
        X : array-like, Series or DataFrame
            Pre-experiment covariates.
        y : array-like
            Target metric.
        cat_cols : bool or list of str, optional
            See :meth:`fit`.
        cat_encoder : callable, optional
            See :meth:`fit`.
        excluded_categories : dict of {str: list}, optional
            See :meth:`fit`.
        scaler : estimator instance, optional
            See :meth:`fit`.
        no_scale_cols : list of str, optional
            See :meth:`fit`.
        fit_kwargs : dict, optional
            See :meth:`fit`.

        Returns
        -------
        numpy.ndarray
            In-sample predictions, shape ``(n_samples,)``.
        """
        self.fit(
            X,
            y,
            cat_cols=cat_cols,
            cat_encoder=cat_encoder,
            excluded_categories=excluded_categories,
            scaler=scaler,
            no_scale_cols=no_scale_cols,
            fit_kwargs=fit_kwargs
        )
        return self.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities for ``X`` (classifier models only).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self.pipe is None:
            raise ValueError('Model is not fitted')
        X = self._to_frame(X)
        return self.pipe.predict_proba(X)

    def fit_predict_proba(
        self,
        X: Any,
        y: Any,
        cat_cols: Optional[Union[bool, List[str]]] = None,
        cat_encoder: Optional[Callable] = None,
        excluded_categories: Optional[Dict[str, List]] = None,
        scaler: Optional[Any] = None,
        no_scale_cols: Optional[List[str]] = None,
        fit_kwargs: Optional[Dict] = None
    ) -> np.ndarray:
        """Fit on ``(X, y)`` and return in-sample class probabilities.

        Equivalent to calling :meth:`fit` followed by :meth:`predict_proba`.
        Intended for classifier models (e.g. ``model='log'``).  All parameters
        are forwarded to :meth:`fit` unchanged.

        Parameters
        ----------
        X : array-like, Series or DataFrame
            Pre-experiment covariates.
        y : array-like
            Binary target metric.
        cat_cols : bool or list of str, optional
            See :meth:`fit`.
        cat_encoder : callable, optional
            See :meth:`fit`.
        excluded_categories : dict of {str: list}, optional
            See :meth:`fit`.
        scaler : estimator instance, optional
            See :meth:`fit`.
        no_scale_cols : list of str, optional
            See :meth:`fit`.
        fit_kwargs : dict, optional
            See :meth:`fit`.

        Returns
        -------
        numpy.ndarray
            Class-probability matrix, shape ``(n_samples, n_classes)``.
        """
        self.fit(
            X,
            y,
            cat_cols=cat_cols,
            cat_encoder=cat_encoder,
            excluded_categories=excluded_categories,
            scaler=scaler,
            no_scale_cols=no_scale_cols,
            fit_kwargs=fit_kwargs
        )
        return self.predict_proba(X)

    def score(
        self,
        X: Any,
        y: Any,
        plot: bool = True,
        predict_proba: bool = False,
    ) -> Dict[str, float]:
        """Report variance-reduction diagnostics for the fitted predictor.

        Computes the residual metric ``y - ŷ`` (or ``y - P(y=1|X)`` when
        ``predict_proba=True``) and summarises how much variance was removed
        relative to the raw metric.

        Parameters
        ----------
        X : array-like, Series or DataFrame
            Features to evaluate on (same format as used in :meth:`fit`).
        y : array-like
            Observed target values.
        plot : bool, default ``True``
            If ``True``, draw a bar chart comparing raw vs. adjusted variance.
        predict_proba : bool, default ``False``
            Use predicted positive-class probabilities instead of point
            predictions when forming the residual.  Requires a classifier
            model.

        Returns
        -------
        dict
            ``var_reduction``
                Fraction of variance removed: ``1 - Var(residual) / Var(y)``.
            ``resid_std``
                Standard deviation of the adjusted (residual) metric.
            ``theta``
                OLS slope coefficient (meaningful for :class:`CUPED` only;
                ``nan`` for all other models).
        """
        y = np.asarray(y, dtype=float)
        if predict_proba:
            adj = y - self.predict_proba(X)[:, 1]
        else:
            adj = y - self.predict(X)
        var_y, var_adj = y.var(), adj.var()
        vr = float(1 - var_adj / var_y) if var_y > 0 else 0.0
        if plot:
            plot_comparison_bars(
                baseline=var_y, values=[var_adj], labels=[self.name],
                title=f'{self.name} Performance', ylabel='Variance', figsize=(5, 3.5),
            )
        theta = float(getattr(self, 'theta_', float('nan')))
        return {
            'var_reduction': vr,
            'resid_std': float(adj.std()),
            'theta': theta,
        }
    
class CUPED(CUPAC):
    """CUPED variance reduction with a single pre-experiment covariate.

    A special case of :class:`CUPAC` that fits an OLS regression on exactly
    one covariate and returns the residuals ``y - θ·X``.  The adjustment
    reduces variance while keeping the treatment-effect estimate unbiased.

    Parameters
    ----------
    name : str, optional
        Name of the regression step in the pipeline.  Defaults to
        ``'CUPED'``.

    Attributes
    ----------
    theta_ : float
        Fitted OLS slope coefficient.  Available after :meth:`fit`.

    Notes
    -----
    Categorical covariates and scaling are intentionally unsupported —
    CUPED is defined for a single *numeric* pre-experiment variable.
    For multi-covariate or categorical adjustments use :class:`CUPAC` or
    :class:`PostStratification` directly.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        name = name or 'CUPED'
        super().__init__('ols', name)

    def _check_X(self, X: Any) -> np.ndarray:
        """Coerce ``X`` to a 2-D single-column array and validate its shape.

        Accepts 1-D arrays, :class:`pandas.Series`, or single-column
        DataFrames / 2-D arrays.

        Parameters
        ----------
        X : array-like or Series
            The covariate to validate.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples, 1)``.

        Raises
        ------
        ValueError
            If ``X`` has more than one column after reshaping.
        """
        if X.ndim == 1:
            if isinstance(X, pd.Series):
                X = X.values
            X = X.reshape(-1, 1)
        if X.shape[1] > 1:
            raise ValueError('X should be one-dimensional')
        return X

    def fit(self, X: Any, y: Any, **kwargs) -> 'CUPED':
        """Fit the OLS predictor on a single covariate ``X`` and target ``y``.

        Parameters
        ----------
        X : array-like or Series
            Single pre-experiment covariate, shape ``(n_samples,)`` or
            ``(n_samples, 1)``.
        y : array-like
            Target metric to adjust, shape ``(n_samples,)``.
        **kwargs
            Accepted for API compatibility with :meth:`CUPAC.fit_predict` but
            ignored — CUPED always uses plain OLS on a single numeric
            covariate.

        Returns
        -------
        CUPED
            The fitted instance (``self``).

        Raises
        ------
        ValueError
            If ``X`` has more than one column.
        """
        X = self._check_X(X)
        result = super().fit(X, y)
        # Extract the OLS theta coefficient so score() can report it.
        self.theta_ = float(self.pipe.named_steps[self.name].coef_[0])
        return result

    def predict(self, X: Any) -> np.ndarray:
        """Predict ``ŷ = θ·X`` using the fitted OLS model.

        Normalises ``X`` through :meth:`_check_X` so Series, 1-D arrays and
        single-column DataFrames are handled consistently with :meth:`fit`.

        Parameters
        ----------
        X : array-like or Series
            Single pre-experiment covariate.

        Returns
        -------
        numpy.ndarray
            Predicted values, shape ``(n_samples,)``.
        """
        return super().predict(self._check_X(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Not meaningful for CUPED (OLS); kept for API compatibility.

        Parameters
        ----------
        X : array-like or Series
            Single pre-experiment covariate.

        Returns
        -------
        numpy.ndarray
            Class-probability matrix forwarded from the parent pipeline.
        """
        return super().predict_proba(self._check_X(X))

    def score(
        self,
        X: Any,
        y: Any,
        plot: bool = True,
        predict_proba: bool = False,
    ) -> dict:
        """Report variance-reduction diagnostics, normalising ``X`` first.

        Delegates to :meth:`CUPAC.score` after passing ``X`` through
        :meth:`_check_X`.

        Parameters
        ----------
        X : array-like or Series
            Single pre-experiment covariate.
        y : array-like
            Observed target values.
        plot : bool, default ``True``
            Draw a variance-comparison bar chart.
        predict_proba : bool, default ``False``
            Use class probabilities instead of point predictions.

        Returns
        -------
        dict
            See :meth:`CUPAC.score` for key descriptions.
        """
        return super().score(self._check_X(X), y, plot=plot, predict_proba=predict_proba)

    def cuped(self, X: Any, y: Any, fit: bool = True) -> np.ndarray:
        """Return the CUPED-adjusted metric ``y - θ·X``.

        Parameters
        ----------
        X : array-like or Series
            Single pre-experiment covariate.
        y : array-like
            Target metric to adjust.
        fit : bool, default ``True``
            If ``True``, (re)fit the model on ``(X, y)`` before adjusting;
            if ``False``, use the already-fitted model.

        Returns
        -------
        numpy.ndarray
            The variance-reduced (adjusted) metric, shape ``(n_samples,)``.
        """
        if fit:
            self.fit(X, y)
        return y - self.predict(X)


class PostStratification(CUPAC):
    """Post-stratification adjustment via OLS on stratum indicator covariates.

    A :class:`CUPAC` variant that removes between-strata variance from the
    metric after the experiment has run.  Categorical and string stratum
    labels are automatically one-hot encoded, so the raw output of
    ``pd.qcut(..., labels=[...])`` can be passed in directly without any
    manual preprocessing.

    Parameters
    ----------
    name : str, optional
        Name of the regression step in the pipeline.  Defaults to
        ``'PostStratification'``.
    fit_intercept : bool, default ``True``
        Whether to fit an intercept in the underlying
        :class:`~sklearn.linear_model.LinearRegression`.  When ``True``
        one dummy column is dropped automatically (``drop='first'``) to
        avoid multicollinearity.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        fit_intercept: bool = True
    ) -> None:
        name = name or 'PostStratification'
        model = LinearRegression(fit_intercept=fit_intercept, copy_X=False)
        super().__init__(model, name)

    def fit(
        self,
        X: Any,
        y: Any,
        cat_cols: Optional[Union[bool, List[str]]] = None,
        cat_encoder: Optional[Callable] = None,
        excluded_categories: Optional[Dict[str, List]] = None,
        scaler: Optional[Any] = None,
        no_scale_cols: Optional[List[str]] = None,
        fit_kwargs: Optional[Dict] = None
    ) -> 'PostStratification':
        """Fit post-stratification OLS, auto-encoding categorical/string columns.

        If ``X`` contains object, string, or
        :class:`~pandas.CategoricalDtype` columns **and** ``cat_encoder`` is
        ``None``, a :class:`~sklearn.preprocessing.OneHotEncoder` is created
        automatically (``drop='first'`` when the model uses an intercept) so
        that string stratum labels such as ``'decile_1'`` work out of the box
        without any manual preprocessing.

        Parameters
        ----------
        X : array-like, Series or DataFrame
            Stratum covariates (numeric or categorical/string).
        y : array-like
            Target metric to adjust.
        cat_cols : bool or list of str, optional
            Categorical columns to encode: ``True`` to auto-detect all
            object/string/categorical columns, or an explicit list of names.
            Defaults to ``True`` when ``X`` contains such columns.
        cat_encoder : callable, optional
            Custom encoder factory accepting a ``categories`` keyword
            argument.  Overrides the default OHE when provided.
        excluded_categories : dict of {str: list}, optional
            Category levels to exclude per column before encoding.
        scaler : estimator instance, optional
            Scaler for numeric columns; ``None`` means passthrough.
        no_scale_cols : list of str, optional
            Numeric columns to skip scaling.

        Returns
        -------
        PostStratification
            The fitted instance (``self``).
        """
        X_frame = self._to_frame(X)
        has_cat = any(
            pd.api.types.is_object_dtype(X_frame[c])
            or isinstance(X_frame[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(X_frame[c])
            for c in X_frame.columns
        )
        if has_cat and cat_encoder is None:
            drop = None
            if getattr(self._model, 'fit_intercept', False):
                drop = 'first'
            cat_cols = cat_cols if cat_cols is not None else True
            cat_encoder = lambda categories: OneHotEncoder(
                categories=categories,
                drop=drop,
                sparse_output=False,
                handle_unknown='ignore'
            )
        return super().fit(
            X, y,
            cat_cols=cat_cols,
            cat_encoder=cat_encoder,
            excluded_categories=excluded_categories,
            scaler=scaler,
            no_scale_cols=no_scale_cols,
            fit_kwargs=fit_kwargs
        )
