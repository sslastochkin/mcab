import warnings
from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Union, Callable, Literal
from numpy.typing import NDArray
from joblib import Parallel, delayed

import pandas as pd
import numpy as np
from scipy import stats

from tqdm.auto import tqdm
from .utils import ProgressParallel, _worker_init_warnings
from .mstats import CORRECTION_METHODS, multiple_pvalue_correction
from .effects import (
    CustomEffectSizer,
    FunctionEffectSizer,
    _EFFECTS_DICT
)
from .plots import (
    plot_aa_pvalues,
    plot_power_cum_means,
    plot_power_curve_data,
    plot_comparison_bars,
    plot_aa_benchmark,
    plot_ab_benchmark,
    plot_multi_test_fwer,
    plot_multi_test_power,
    plot_directional_analysis
)
from .transform import Linearizer

__all__ = [
    'RandomData',
    'AaDataDept1s',
    'AaDataIid',
    'AaDataRatio',
    'DesignerIid',
    'DesignerRatio',
    'DesignerRatioLin',
    'DesignerDept1s',
    'BenchMarker',
]

def proportion_ci_wilson(count, n, alpha=0.05):
    """
    Wilson score confidence interval for proportion.
    
    A more accurate method for calculating the confidence interval of a proportion,
    especially for small samples and extreme values of proportions (close to 0 or 1).
    
    Parameters
    ----------
    count : int
        Number of successes (events).
    n : int
        Total number of trials.
    alpha : float, default=0.05
        Significance level for the confidence interval.
    
    Returns
    -------
    p : float
        Observed proportion (count / n).
    left_bound : float
        Lower bound of the confidence interval.
    right_bound : float
        Upper bound of the confidence interval.
    
    References
    ----------
    Wilson, E. B. (1927). "Probable inference, the law of succession,
    and statistical inference". Journal of the American Statistical Association.
    """
    p = count / n
    z = stats.norm.ppf(1 - alpha / 2)
    
    # Wilson formula
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    
    left_bound = center - margin
    right_bound = center + margin
    
    # Clip bounds to [0, 1]
    left_bound = max(0.0, min(1.0, left_bound))
    right_bound = max(0.0, min(1.0, right_bound))
    
    return p, left_bound, right_bound

def first_type_error_ci(pvalue_vector, alpha=0.05):
    """
    Confidence interval for first type error.
    
    Calculates confidence interval for the proportion of type I errors
    in a series of statistical tests using the Wilson score method.
    
    Parameters
    ----------
    pvalue_vector : array-like
        Vector of p-values from statistical tests.
    alpha : float, default=0.05
        Significance level for both the test and the confidence interval.
    
    Returns
    -------
    p : float
        Observed proportion of p-values < alpha.
    left_bound : float
        Lower bound of the confidence interval.
    right_bound : float
        Upper bound of the confidence interval.
    """
    pvalue_vector = np.asarray(pvalue_vector).flatten()
    n = len(pvalue_vector)
    count = np.sum(pvalue_vector < alpha)
    
    return proportion_ci_wilson(count, n, alpha)

def mean_ci_normal(values, alpha=0.05):
    """
    Returns mean and its normal confidence interval.
    """
    n = len(values)
    p = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(n)
    z = stats.norm.ppf(1 - alpha / 2)
    left_bound, right_bound = (p - z * se, p + z * se)
    return p, left_bound, right_bound

def cum_mean(data):
    """
    Calculates cumulative mean.
    """
    return np.cumsum(data) / np.arange(1, len(data) + 1)

class RandomData:
    """
    Reproducible synthetic data generator for A/B-test simulations.

    Thin convenience wrapper around :class:`numpy.random.Generator`
    (``np.random.default_rng``) that produces the kinds of samples used by the
    Designer/BenchMarker classes in this module: simple i.i.d. metrics as well
    as ratio metrics (numerator/denominator pairs).

    All draws come from a single seeded generator stored on the instance, so a
    given ``seed`` fully determines every method's output and makes simulations
    reproducible. Note that the generator carries state: consecutive calls
    advance the stream and therefore return different samples. Re-create the
    object (or use the same ``seed``) to reproduce a specific sequence.

    Parameters
    ----------
    seed : int, default=42
        Seed for the underlying ``np.random.default_rng`` generator.

    Attributes
    ----------
    rng : numpy.random.Generator
        The seeded random generator backing every method.
    seed : int
        The seed the instance was created with (kept for ``repr`` / reference).

    Methods
    -------
    i.i.d. metrics (return a single 1-D ``ndarray``):
        proportion_data(size, p)
            Bernoulli 0/1 samples (e.g. conversion flags).
        normal_data(size, mean, std)
            Gaussian samples (e.g. continuous KPIs).
        uniform_data(size, low, high)
            Uniform samples on ``[low, high)``.
        exponential_data(size, scale)
            Exponential samples (e.g. heavy-tailed durations/revenue).
        poisson_data(size, lam)
            Poisson counts (e.g. number of events).
    ratio metrics (return a ``(numerator, denominator)`` tuple of ``ndarray``):
        ratio_data_ctr(...)
            CTR-like data with ``numerator <= denominator``
            (clicks / impressions).
        ratio_data_avg_bill(...)
            Average-check-like data (revenue / purchases).

    Examples
    --------
    >>> rd = RandomData(seed=42)
    >>> rd.normal_data(size=5, mean=100, std=10)        # i.i.d. metric
    >>> clicks, shows = rd.ratio_data_ctr(size=1000)     # ratio metric
    >>> RandomData(seed=42).normal_data(size=3)          # reproducible
    """

    def __init__(self, seed=42):
        self.rng  = np.random.default_rng(seed)
        self.seed = seed
    
    def __repr__(self):
        return f"RandomData(seed={self.seed})"

    def proportion_data(self, size=10_000, p=0.5):
        return self.rng.binomial(n=1, p=p, size=size)

    def normal_data(self, size=10_000, mean=100, std=30):
        return self.rng.normal(loc=mean, scale=std, size=size)
    
    def uniform_data(self, size=10_000, low=0, high=1):
        return self.rng.uniform(low=low, high=high, size=size)
    
    def exponential_data(self, size=10_000, scale=1_000):
        return self.rng.exponential(scale=scale, size=size)
    
    def poisson_data(self, size=10_000, lam=1):
        return self.rng.poisson(lam=lam, size=size)
    
    def ratio_data_ctr(
        self,
        size: int = 10_000,
        ctr: float = 0.05,
        mean_events_per_user: float = 10.0,
        std_events: float = 3.0
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Generate ratio metric data for CTR-like metrics (numerator <= denominator).
        
        Simulates user-level data where:
        - Denominator: number of events (impressions, views, etc.)
        - Numerator: number of successes (clicks, conversions, etc.)
        - Constraint: numerator <= denominator
        
        Parameters
        ----------
        size : int, default=10_000
            Number of users.
        ctr : float, default=0.05
            Target click-through rate (conversion rate).
        mean_events_per_user : float, default=10.0
            Average number of events per user.
        std_events : float, default=3.0
            Standard deviation of events per user.
        
        Returns
        -------
        numerator : ndarray
            Number of successes per user (clicks, conversions).
        denominator : ndarray
            Number of events per user (impressions, views).
        
        Examples
        --------
        >>> rd = RandomData(seed=42)
        >>> clicks, impressions = rd.ratio_data_ctr(size=1000, ctr=0.05)
        >>> observed_ctr = clicks.sum() / impressions.sum()
        >>> print(f"Observed CTR: {observed_ctr:.4f}")
        """
        # Generate denominators (events per user) - must be positive
        denominator = np.maximum(
            1,
            self.rng.normal(mean_events_per_user, std_events, size=size).astype(int)
        )
        
        # Generate numerators (successes) - binomial with p=ctr
        numerator = self.rng.binomial(n=denominator, p=ctr)
        
        return numerator.astype(int), denominator.astype(int)
    
    def ratio_data_avg_bill(
        self,
        size: int = 10_000,
        scale: float = 1500.0,
        lam: float = 2.0
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        """
        numerator   = self.rng.exponential(scale, size=size)
        denominator = self.rng.poisson(lam=lam, size=size) + 1
        return numerator, denominator

class _DesignerBase:
    """
    Serice class with service functions for Designers.
    """

    def validated_n_controls(self, n_controls):
        # Get data length - works for both AaDataIid and AaDataRatio
        n = len(self.target.idx)
        assert n >= 20, 'data must have at least 20 observations'
        
        if n_controls is None:
            n_controls = n // 2
        elif n_controls < 0:
            raise ValueError('n_controls must be greater than 0')
        elif n_controls > 0 and n_controls < 1:
            n_controls = round(n * n_controls)

        assert n_controls <= (n - 10), 'test group must have at least 10 observations (n_controls too large)'
        assert n_controls >= 10, 'control group must have at least 10 observations.'

        return n_controls
    
    def aa_sims(
        self,
        pval_func: Callable = None,
        n_sims: int = 5_000,
        control_size: Optional[Union[int, float]] = None,
        n_jobs: int = 1,
        verbose: bool = True,
        plot: bool = True,
        figsize: Optional[tuple] = None
    ):
        """
        Returns p-values vector for n_sims AA-simulations.

        Arguments:
        ---
        - pval_func: any function, should return p-value. if not specified - self.pval_func will be used
        - n_sims: int - number of simulations
        - control_size: Optional[Union[int, float]] - number of controls or ratio of control data
        - n_jobs: int - number of jobs
        - verbose: bool - verbose mode
        - plot: bool - plot p-values distribution
        - figsize: Optional[tuple] - figure size (width, height)

        Returns:
        ---
        - p_values: np.array - p-values vector
        """
        pval_func = pval_func or self.pval_func
        if pval_func is None:
            raise ValueError('pval_func must be specified.')
        
        n_controls = self.validated_n_controls(control_size)

        if verbose:
            ParallelClass = ProgressParallel(
                use_tqdm=True,
                total=n_sims,
                desc='Simulations',
                unit='sim',
                leave=True,
                n_jobs=n_jobs,
                initializer=_worker_init_warnings
            )
        else:
            ParallelClass = Parallel(n_jobs=n_jobs, initializer=_worker_init_warnings)

        p_values = ParallelClass(
            delayed(self._single_sim)(i, pval_func, n_controls, n_jobs, None)
            for i in range(n_sims)
        )

        p_values = np.array(p_values)
        if plot:
            # Calculate cumulative mean of errors
            errors = (p_values < self.alpha).astype(int)
            cumulative_mean = cum_mean(errors)
            
            plot_aa_pvalues(
                pvals=p_values,
                alpha=self.alpha,
                mean_ci_func=mean_ci_normal,
                first_type_error_ci_func=first_type_error_ci,
                cumulative_mean=cumulative_mean,
                ax=None,
                figsize=figsize
            )

        return p_values
    
    def calculate_some_power(
        self,
        effect: float,
        pval_func: Optional[Callable] = None,
        control_size: Optional[Union[int, float]] = None,
        n_sims: int = 5_000,
        n_jobs: int = 1,
        verbose: bool = True,
        as_rejection: bool = True
    ):
        """
        Makes n_sims simulations of test power.

        Parameters
        ----------
        as_rejection : bool, default=True
            If True, returns a vector of 0 and 1, where 1 = test discovered an
            effect at self.alpha significance level, 0 otherwise. If False,
            returns the raw p-values vector instead of the rejection indicators
            (useful when the caller needs the p-values themselves).

        Returns
        -------
        np.array
            Vector of rejection indicators (0/1) if ``as_rejection=True``,
            otherwise the raw p-values vector.
        """
        pval_func = pval_func or self.pval_func
        if pval_func is None:
            raise ValueError('pval_func must be specified.')
        
        n_controls = self.validated_n_controls(control_size)

        if verbose:
            ParallelClass = ProgressParallel(
                use_tqdm=True,
                total=n_sims,
                desc='Simulations',
                unit='sim',
                leave=False,
                n_jobs=n_jobs,
                initializer=_worker_init_warnings
            )
        else:
            ParallelClass = Parallel(n_jobs=n_jobs, initializer=_worker_init_warnings)

        results = ParallelClass(
            delayed(self._calculate_some_single_power)(i, pval_func, n_controls, effect, n_jobs, None, as_rejection)
            for i in range(n_sims)
        )

        results = np.array(results)
        return results
    
    def calculate_power_curve(
        self,
        pval_funcs: Optional[List[Tuple[str, Callable]]] = None,
        effects_list: Optional[List[float]] = None,
        control_size: Optional[Union[int, float]] = None,
        n_sims: int = 5_000,
        n_jobs: int = 1,
        verbose: bool = True,
        plot: bool = True,
        annotate_curve: bool = True,
        target_power: Optional[float] = None,
        figsize: Optional[tuple] = None
    ):
        """
        Builds power curve for given effects list.
        
        Parameters
        ----------
        target_power : float, optional
            Target power for the horizontal reference line on the plot.
            If None, defaults to ``1 - self.beta``.
        figsize : tuple, optional
            Figure size (width, height) for the plot.
        """
        if pval_funcs is None:
            if self.pval_func is None:
                raise ValueError('pval_funcs must be specified.')
            pval_funcs = [('self_func', self.pval_func)]
        
        if effects_list is None:
            effects_list = np.linspace(0.01, 0.10, 10)

        n_controls = self.validated_n_controls(control_size)

        effects_cnt = len(effects_list)
        funcs_cnt   = len(pval_funcs)
        elems_cnt   = effects_cnt * funcs_cnt

        power_curves = []
        power_cummeans = []

        if verbose:
            print('Calculating power curve...')
        for fn, (func_name, pval_func) in enumerate(pval_funcs):
            power_curve_fn = np.full(effects_cnt, np.nan)
            cummeans_fn    = []
            for en, effect in enumerate(effects_list):
                curr_i = (fn+1) * (en+1)
                power_data = self.calculate_some_power(
                    pval_func=pval_func,
                    effect=effect,
                    control_size=n_controls,
                    n_sims=n_sims,
                    n_jobs=n_jobs,
                    verbose=verbose
                )
                cummeans_fn.append(cum_mean(power_data))
                power = power_data.mean()
                power_curve_fn[en] = power
                if verbose:
                    print(f' - [{curr_i}/{elems_cnt}] {effect:,.4f}: {power*100:.2f}% {func_name}')
            power_curves.append((func_name, power_curve_fn))
            power_cummeans.append(cummeans_fn)
        if target_power is None:
            target_power = 1 - self.beta
        if plot:
            plot_power_curve_data(effects_list, power_curves, power_cummeans, target_power=target_power, annotate=annotate_curve, figsize=figsize)
        return power_curves
    
    def find_mde(
        self,
        pval_func: Optional[Callable] = None,
        target_power: Optional[float] = None,
        max_effect: float = 1.0,
        control_size: Optional[Union[int, float]] = None,
        tolerance: float = 0.005,  # reached power tolerance 0.005=±0.5%
        n_sims: int = 5_000,
        n_jobs: int = 1,
        plot: bool = True,
        verbose: bool = True,
        figsize: Optional[tuple] = None
    ):
        """
        Finds Minimal Detectable Effect (MDE) for given power.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height) for the plot.
        """

        if pval_func is None:
            if self.pval_func is None:
                raise ValueError('pval_func must be specified.')
            pval_func = self.pval_func
        
        if target_power is None:
            target_power = 1 - self.beta

        n_controls = self.validated_n_controls(control_size)

        # Check: target power is reachable
        power_at_max = self.calculate_some_power(
            pval_func=pval_func,
            effect=max_effect,
            control_size=n_controls,
            n_sims=n_sims,
            n_jobs=n_jobs,
            verbose=False
        ).mean()

        if power_at_max < target_power:
            if verbose:
                warnings.warn(f'⚠️ Target power {target_power:.2f} is not reachable. Maximum power for max_effect={max_effect} is {power_at_max:.3f}')
            res = {
                'mde': max_effect,
                'mde_power': power_at_max,
                'best_power_data': None,
                'iterations': 0,
                'target_reached': False
            }
            return res

        # Binary search for MDE
        left = 0.001
        right = max_effect
        
        best_mde = None
        best_power = None
        best_power_data = None
        best_diff = float('inf')  # tracking minimal deviation from goal
        
        iterations = 0
        
        while right - left > 1e-16:
            iterations += 1
            mid = (left + right) / 2
            
            power_data = self.calculate_some_power(
                pval_func=pval_func,
                effect=mid,
                control_size=n_controls,
                n_sims=n_sims,
                n_jobs=n_jobs,
                verbose=False
            )
            power = power_data.mean()
            
            # Tracking BEST approx to target power
            diff = abs(power - target_power)
            if diff < best_diff:
                best_diff = diff
                best_mde = mid
                best_power = power
                best_power_data = power_data  # save only the best
            
            # Check: did we reach the target power?
            if diff < tolerance:
                if verbose:
                    print(f'✓ Target power reached for {iterations} iterations.')
                break
            elif power < target_power:
                # Need bigger effect
                left = mid
            else:
                # Need smaller effect
                right = mid
        else:
            if verbose:
                print('x Target power not reached.')

        res = {
            'mde': best_mde,
            'mde_power': best_power,
            'best_power_data': best_power_data,
            'iterations': iterations,
            'target_reached': abs(best_power - target_power) < tolerance
        }
        
        if best_power_data is not None:
            best_power_data_cum = cum_mean(best_power_data)
            if verbose:
                status = '✓' if res['target_reached'] else 'x'
                print(f'{status} MDE: {best_mde:.4f}; Power: {best_power:.4f}; '
                    f'Target: {target_power:.2f}; Iterations: {iterations:,}')
            if plot:
                plot_power_cum_means(
                    effects_list=[best_mde],
                    cummeans=[best_power_data_cum],
                    target_power=target_power,
                    ax=None,
                    figsize=figsize
                )

        return res

def _take_rows(obj, idx):
    """Row-index a 2-D ndarray / DataFrame preserving its original type.

    The user supplies a custom transformer and we must hand the covariates back
    to it as the SAME type they passed in (e.g. a DataFrame stays a DataFrame),
    so positional integer indexing must go through ``.iloc`` for a DataFrame.
    """
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[idx]
    return obj[idx]


def _concat_rows(a, b):
    """Concatenate two covariate chunks preserving type (2-D ndarray / DataFrame)."""
    if isinstance(a, pd.DataFrame):
        return pd.concat([a, b], axis=0)
    return np.concatenate([a, b])


class AaDataDept1s:
    """
    Container for dependent one-sample data with an artificial (synthetic)
    control, used for Monte-Carlo simulations.

    The modeled metric is the per-unit difference between the observed test
    data and its synthetic control (``data_test - data_synt``).
    """
    def __init__(
        self,
        data_test: Union[pd.Series, NDArray],
        data_synt: Union[pd.Series, NDArray]
    ):
        data_raw  = np.array(data_test)
        data_synt = np.array(data_synt)
        self.data_mod = data_raw - data_synt
        self.data_raw = data_raw
        self.transformer = None
        self.covariates = None
        self.idx = np.arange(len(data_raw))

    def __repr__(self):
        return f'AaDataDept1s()'

class AaDataIid:
    """
    Class for pre-experiment independent (user-aggregated) data using for simulations.
    Supports data transformations like cuped, cupac, etc (if given)
    """
    def __init__(
        self,
        data: Union[pd.Series, np.array],
        transformer: Optional[any] = None,
        covariates: Optional[Union[pd.Series, np.array]] = None,
        proportion: bool = False
    ):
        
        # Initial checks
        
        if (transformer is not None) or (covariates is not None):
            assert (transformer is not None) and (covariates is not None), 'Both data transformer covariates must be specified if one is given.'

        assert data.ndim == 1,  'Data must be 1-dimensional'
        assert len(data) >= 20, 'Data must have at least 20 elements'

        # Attributes assignment

        self.data_raw = np.array(data)
        self.idx = np.arange(len(self.data_raw))

        self.transformer = transformer
        if covariates is not None:
            # Keep covariates in the EXACT type the user provided (ndarray /
            # DataFrame / Series) so the custom transformer receives what it
            # expects; do not coerce to a bare ndarray.
            assert covariates.shape[0] == self.data_raw.shape[0], \
                'data_transformer_covariates must have the same length as data'
            self.covariates = covariates
        else:
            self.covariates = None

        # Proportion warning
        self.proportion = proportion
        if self.data_raw.min() >= 0 and self.data_raw.max() <= 1 and not proportion:
            warnings.warn('Given proportion=False but target between 0 and 1. Are you shure?')

        if proportion:
            if not np.all((self.data_raw == 0) | (self.data_raw == 1)):
                raise ValueError(
                    'Given proportion=True but data contains values other than 0 and 1. '
                    f'Unique values found: {np.unique(self.data_raw)}'
                )
    
    def __repr__(self):
        n = len(self.data_raw)
        prop_str = ", proportion=True" if self.proportion else ""
        return f"AaDataIid(n={n}{prop_str})"
    
class AaDataRatio:
    """
    Class for pre-experiment ratio metric data (numerator/denominator pairs).
    Supports data transformations like linearization, delta method, etc (if given).
    
    For ratio metrics like CTR (clicks/impressions), average check (revenue/purchases), etc.
    """
    def __init__(
        self,
        data: Tuple[NDArray, NDArray],
        transformer: Optional[Callable] = None,
        covariates: Optional[Union[NDArray, pd.DataFrame]] = None,
        proportion: bool = False
    ):
        """
        Initialize ratio metric data.

        Parameters
        ----------
        data : tuple of (numerator, denominator)
            Numerator values (e.g. clicks, revenue) and denominator values
            (e.g. impressions, purchases).
        transformer : callable, optional
            Single variance-reduction function. It receives the SINGLE
            ``covariates`` object together with the pooled numerator and
            denominator and returns the adjusted ``(num, denom)`` pair:
            ``transformer(covariates, num, denom) -> (num_adj, denom_adj)``.
        covariates : 2-D ndarray or DataFrame, optional
            A SINGLE 2-D covariate matrix (no longer a per-numerator/denominator
            tuple, and a 1-D vector is rejected). It is passed to
            ``transformer`` unchanged, preserving its type, so a DataFrame stays
            a DataFrame.
        proportion : bool, default False
            Whether this is a proportion metric (CTR-like) where numerator <= denominator.
        """
        # Initial checks
        numerator, denominator = data
        
        numerator   = np.array(numerator)
        denominator = np.array(denominator)
        
        assert numerator.ndim == 1, 'Numerator must be 1-dimensional'
        assert denominator.ndim == 1, 'Denominator must be 1-dimensional'
        assert len(numerator) == len(denominator), 'Numerator and denominator must have the same length'
        assert len(numerator) >= 20, 'Data must have at least 20 elements'
        
        # Proportion validation
        self.proportion = proportion
        if proportion:
            assert np.all(numerator >= 0), 'For proportion metrics, numerator must be non-negative'
            assert np.all(denominator >= 0), 'For proportion metrics, denominator values must be non-negative'
            assert np.all(numerator <= denominator), 'For proportion metrics, numerator must be <= denominator'
        
        # Attributes assignment
        self.data_raw = (numerator, denominator)
        self.idx = np.arange(len(numerator))

        if (transformer is not None) or (covariates is not None):
            assert (transformer is not None) and (covariates is not None), \
                'Both transformer and covariates must be specified if one is given.'

        self.transformer = transformer
        if covariates is not None:
            # Accept Series and 1-D arrays by reshaping to (n, 1) / single-column
            # DataFrame so that CUPED-style transformers that take a single
            # pre-experiment covariate work without the user having to reshape.
            if isinstance(covariates, pd.Series):
                covariates = covariates.to_frame()
            elif isinstance(covariates, np.ndarray) and covariates.ndim == 1:
                covariates = covariates.reshape(-1, 1)
            assert isinstance(covariates, (np.ndarray, pd.DataFrame)), \
                'covariates must be an ndarray, DataFrame, or Series.'
            assert covariates.ndim == 2, \
                'covariates must be 2-dimensional after coercion.'
            assert covariates.shape[0] == numerator.shape[0], \
                'covariates must have the same length as data'
            self.covariates = covariates
        else:
            self.covariates = None
    
    def __repr__(self):
        n = len(self.data_raw[0])
        prop_str = ", proportion=True" if self.proportion else ""
        return f"AaDataRatio(n={n}{prop_str})"

class DesignerIid(_DesignerBase):
    """
    Class for A/B tests iid (user-aggregated) design.

    Arguments:
    ---
    - target: AaDataIid or array-like - target metric data. If not AaDataIid, will be wrapped automatically.
    - alpha: float - significance level
    - seed: int - seed for random data generation
    - name: str - designer name
    - pval_func: callable - function to calculate p-value
    """
    def __init__(
        self,
        target: Union[AaDataIid, pd.Series, np.ndarray],
        alpha: float = 0.05,
        beta: float = 0.20,
        seed: int = 42,
        name: Optional[str] = None,
        pval_func: Optional[Callable] = None,
        effect_sizer: Literal['percent', 'const', 'custom', 'function'] = 'percent',
        custom_data: NDArray = None,
        raw_data: NDArray = None,
        func: Optional[Callable] = None
    ):
        # If target is not an AaDataIid, build it under the hood
        if not isinstance(target, AaDataIid):
            data = np.array(target)
            # Automatically detect whether the data is a proportion
            is_proportion = (data.min() >= 0 and data.max() <= 1 and
                           np.all((data == 0) | (data == 1)))
            target = AaDataIid(data, proportion=is_proportion)
        
        self.target = target
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        self.name = name
        self.pval_func = pval_func

        effect_sizer = _EFFECTS_DICT[effect_sizer]
        if effect_sizer == CustomEffectSizer:
            if custom_data is None:
                raise ValueError("custom_data must be provided if effect_sizer is 'custom'")
            if custom_data.shape != target.data_raw.shape:
                raise ValueError("custom_data must have the same shape as target data")
            custom_data = custom_data
            raw_data = target.data_raw
        elif effect_sizer == FunctionEffectSizer:
            if func is None:
                raise ValueError("func must be provided if effect_sizer is 'function'")

        self.add_effect = effect_sizer(
            proportion=target.proportion,
            ratio=False,
            seed=seed,
            func=func,
            custom_data=custom_data,
            raw_data=raw_data
        )
    
    def __repr__(self):
        name_str = f"'{self.name}'" if self.name else "None"
        has_pval = self.pval_func is not None
        return f"DesignerIid(name={name_str}, alpha={self.alpha}, has_pval_func={has_pval}, target={self.target})"

    def _single_sim(
        self,
        seed: int,
        pval_func: Callable,
        n_controls: int,
        n_jobs: int,
        idx: np.array,
        effect: float = 0.0,
        return_rejection: bool = False
    ):
        """
        Single AA/AB simulation.
        
        Parameters
        ----------
        i : int
            Simulation index.
        pval_func : Callable
            Function to calculate p-value.
        n_controls : int
            Number of control observations.
        n_jobs : int
            Number of parallel jobs (affects RNG).
        idx : np.array or None
            Pre-shuffled indices. If None, will shuffle internally.
        effect : float, default=0.0
            Effect size to apply to test group. 0.0 for AA-test.
        return_rejection : bool, default=False
            If True, returns 1/0 for rejection at alpha level. If False, returns p-value.
        
        Returns
        -------
        float or int
            P-value if return_rejection=False, rejection indicator (0/1) if True.
        """

        if idx is None:
            idx = self.target.idx.copy()
            if n_jobs == 1:
                self.rng.shuffle(idx)
            else:
                rng = np.random.default_rng(seed)
                rng.shuffle(idx)

        test_idx    = idx[:n_controls]
        control_idx = idx[n_controls:]

        # Apply effect only (no covariate transform here).
        #
        # The variance-reduction transform (CUPED/CUPAC/...) is intentionally
        # NOT delegated to add_effect: doing so would fit a SEPARATE regression
        # on each group, which both wipes the effect and inflates the type I
        # error. We pass covariates=None and apply the transform once below.
        data = self.target.data_raw
        control = data[control_idx]
        if isinstance(self.add_effect, CustomEffectSizer):
            first_argument_test = test_idx
        elif isinstance(self.add_effect, FunctionEffectSizer):
            first_argument_test = (data[test_idx], control, data, test_idx, control_idx, idx)
        else:
            first_argument_test = data[test_idx]

        test = self.add_effect(
            first_argument_test,
            effect=effect,
            seed=seed
        )

        # Variance reduction (CUPED / CUPAC / etc.).
        #
        # The transformer is fitted ONCE on the pooled test+control sample and
        # the adjusted metric is split back into the two groups. Because the
        # pre-experiment covariate is independent of the random group split, the
        # single shared regression cannot absorb the treatment shift: the
        # difference in group means (and thus the injected effect) is preserved,
        # while the variance is reduced. Crucially, the SAME coefficients are
        # applied to both groups, so the intercept cancels in the test/control
        # comparison and the type I error stays controlled. Fitting the
        # transformer separately per group would instead inflate it.
        transformer = self.target.transformer
        covariates = self.target.covariates
        if transformer is not None and covariates is not None:
            n_test = len(test)
            pooled_target = np.concatenate([test, control])
            # Preserve covariate type (ndarray / DataFrame / Series) for the
            # user-supplied transformer.
            pooled_cov = _concat_rows(
                _take_rows(covariates, test_idx),
                _take_rows(covariates, control_idx)
            )
            pooled_adjusted = transformer(pooled_cov, pooled_target)
            test = pooled_adjusted[:n_test]
            control = pooled_adjusted[n_test:]

        pval = pval_func(test, control)
        
        if return_rejection:
            return int(pval < self.alpha)
        return pval

    def _calculate_some_single_power(
        self,
        seed: int,
        pval_func: Callable,
        n_controls: int,
        effect: float,
        n_jobs: int,
        idx: np.array,
        as_rejection: bool = True
    ):
        """
        Single simulation of power.

        If ``as_rejection=True`` returns 1 if the test discovered an effect at
        the self.alpha significance level, 0 otherwise. If ``as_rejection=False``
        returns the raw p-value instead.

        Delegates to _single_sim with return_rejection=as_rejection.
        """
        return self._single_sim(
            seed=seed,
            pval_func=pval_func,
            n_controls=n_controls,
            n_jobs=n_jobs,
            idx=idx,
            effect=effect,
            return_rejection=as_rejection
        )
    
class DesignerRatio(_DesignerBase):
    """
    Class for A/B tests with ratio metrics design.
    
    Similar to DesignerIid but works with ratio metrics (numerator/denominator pairs).
    
    Arguments:
    ---
    - target: AaDataRatio or tuple/list - target ratio metric data.
      If not AaDataRatio, should be tuple/list of 2 elements: (numerator, denominator)
    - alpha: float - significance level
    - seed: int - seed for random data generation
    - name: str - designer name
    - pval_func: callable - function to calculate p-value
    """
    def __init__(
        self,
        target: Union[AaDataRatio, Tuple, List],
        alpha: float = 0.05,
        beta: float = 0.20,
        seed: int = 42,
        name: Optional[str] = None,
        pval_func: Optional[Callable] = None,
        effect_sizer: Literal['percent', 'const', 'custom', 'function'] = 'percent',
        ratio: Literal['int', 'float'] = 'float',
        custom_data: NDArray = None,
        raw_data: NDArray = None,
        func: Optional[Callable] = None,
    ):
        # Linearized ratio metrics are handled by the dedicated
        # DesignerRatioLin class, so 'lin' is not accepted here.
        if ratio not in ('int', 'float'):
            raise ValueError(
                "DesignerRatio supports ratio in {'int', 'float'}; "
                "use DesignerRatioLin for ratio='lin'."
            )
        # If target is not an AaDataRatio, build it under the hood
        if not isinstance(target, AaDataRatio):
            if isinstance(target, (tuple, list)):
                if len(target) != 2:
                    raise ValueError(f"Target tuple/list must have 2 elements (numerator, denominator), got {len(target)}")
                
                numerator, denominator = target
                numerator = np.array(numerator)
                denominator = np.array(denominator)
                
                # Automatically detect whether the data is a proportion
                is_proportion = (np.all(numerator >= 0) and
                               np.all(numerator <= denominator) and
                               np.all(numerator == numerator.astype(int)) and
                               np.all(denominator == denominator.astype(int)))
                
                target = AaDataRatio((numerator, denominator), proportion=is_proportion)
            else:
                raise ValueError("Target must be AaDataRatio or tuple/list of (numerator, denominator)")
        
        self.target = target
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        self.name = name
        self.pval_func = pval_func
        self.ratio = ratio

        effect_sizer = _EFFECTS_DICT[effect_sizer]
        if effect_sizer == CustomEffectSizer:
            if custom_data is None:
                raise ValueError("custom_data must be provided if effect_sizer is 'custom'")
            if custom_data[0].shape != target.data_raw[0].shape:
                raise ValueError("custom_data must have the same shape as target data")
            custom_data = custom_data
            raw_data = target.data_raw
        elif effect_sizer == FunctionEffectSizer:
            if func is None:
                raise ValueError("func must be provided if effect_sizer is 'function'")

        self.add_effect = effect_sizer(
            proportion=target.proportion,
            ratio=ratio,
            seed=seed,
            func=func,
            custom_data=custom_data,
            raw_data=raw_data
        )
    
    def __repr__(self):
        name_str = f"'{self.name}'" if self.name else "None"
        has_pval = self.pval_func is not None
        return f"DesignerRatio(name={name_str}, alpha={self.alpha}, has_pval_func={has_pval}, target={self.target})"
    
    def _single_sim(
        self,
        seed: int,
        pval_func: Callable,
        n_controls: int,
        n_jobs: int,
        idx: np.array,
        effect: float = 0.0,
        return_rejection: bool = False
    ):
        """
        Single AA/AB simulation for ratio metrics.
        
        Parameters
        ----------
        i : int
            Simulation index.
        pval_func : Callable
            Function to calculate p-value. Should accept (test_num, test_denom, control_num, control_denom).
        n_controls : int
            Number of control observations.
        n_jobs : int
            Number of parallel jobs (affects RNG).
        idx : np.array or None
            Pre-shuffled indices. If None, will shuffle internally.
        effect : float, default=0.0
            Effect size to apply to test group. 0.0 for AA-test.
        return_rejection : bool, default=False
            If True, returns 1/0 for rejection at alpha level. If False, returns p-value.
        
        Returns
        -------
        float or int
            P-value if return_rejection=False, rejection indicator (0/1) if True.
        """
        if idx is None:
            idx = self.target.idx.copy()
            if n_jobs == 1:
                self.rng.shuffle(idx)
            else:
                rng = np.random.default_rng(seed)
                rng.shuffle(idx)
        
        test_idx = idx[:n_controls]
        control_idx = idx[n_controls:]

        num_raw, denom_raw = self.target.data_raw

        # Apply effect only (no covariate transform here).
        #
        # As in DesignerIid, the variance-reduction transform is intentionally
        # NOT delegated to add_effect: fitting a SEPARATE regression per group
        # would both wipe the effect and inflate the type I error. We pass
        # covariates=None and apply the transform once below on the pooled
        # test+control sample.
        control_num, control_denom = num_raw[control_idx], denom_raw[control_idx]

        if isinstance(self.add_effect, CustomEffectSizer):
            first_argument = test_idx
        elif isinstance(self.add_effect, FunctionEffectSizer):
            first_argument = (
                (num_raw[test_idx], denom_raw[test_idx]),
                (control_num, control_denom),
                (num_raw, denom_raw),
                test_idx, control_idx, idx
            )
        else:
            first_argument = (num_raw[test_idx], denom_raw[test_idx])
        test_num, test_denom = self.add_effect(
            first_argument,
            effect=effect,
            seed=seed
        )

        # Variance reduction (CUPED / CUPAC / etc.) for ratio metrics.
        #
        # A SINGLE user transformer receives the SINGLE covariate matrix along
        # with the pooled numerator and denominator and returns the adjusted
        # pair: transformer(covariates, num, denom) -> (num_adj, denom_adj).
        # It is fitted ONCE on the pooled test+control sample and the adjusted
        # values are split back into the two groups, so the SAME coefficients
        # are applied to both. The intercept therefore cancels in the
        # test/control comparison: the difference in group means (and the
        # injected effect) is preserved while the variance is reduced and the
        # type I error stays controlled. The covariate object is passed in the
        # user's original type (ndarray / DataFrame) via the row-index helpers.
        transformer = self.target.transformer
        covariates = self.target.covariates
        if transformer is not None and covariates is not None:
            n_test = len(test_num)

            pooled_num = np.concatenate([test_num, control_num])
            pooled_denom = np.concatenate([test_denom, control_denom])
            pooled_cov = _concat_rows(
                _take_rows(covariates, test_idx),
                _take_rows(covariates, control_idx)
            )

            pooled_num_adj, pooled_denom_adj = transformer(
                pooled_cov, (pooled_num, pooled_denom)
            )

            test_num = pooled_num_adj[:n_test]
            control_num = pooled_num_adj[n_test:]
            test_denom = pooled_denom_adj[:n_test]
            control_denom = pooled_denom_adj[n_test:]

        test_data = (test_num, test_denom)
        control_data = (control_num, control_denom)

        pval = pval_func(test_data, control_data)
        
        if return_rejection:
            return int(pval < self.alpha)
        return pval
    
    def _calculate_some_single_power(
        self,
        seed: int,
        pval_func: Callable,
        n_controls: int,
        effect: float,
        n_jobs: int,
        idx: np.array,
        as_rejection: bool = True
    ):
        """
        Single simulation of power for ratio metrics.

        If ``as_rejection=True`` returns 1 if the test discovered an effect at
        the self.alpha significance level, 0 otherwise. If ``as_rejection=False``
        returns the raw p-value instead.

        Delegates to _single_sim with return_rejection=as_rejection.
        """
        return self._single_sim(
            seed=seed,
            pval_func=pval_func,
            n_controls=n_controls,
            n_jobs=n_jobs,
            idx=idx,
            effect=effect,
            return_rejection=as_rejection
        )


class DesignerRatioLin(_DesignerBase):
    """
    Class for A/B tests on a LINEARIZED ratio metric.

    The ratio metric (numerator / denominator) is collapsed into a single
    linearized per-object value ``num - theta * denom`` (Taylor / delta-method
    linearization), where ``theta`` is the control-group ratio. After
    linearization the metric is a plain 1-D array, so the optional CUPED/CUPAC
    transformer is applied exactly as in DesignerIid: fitted ONCE on the pooled
    test+control linearized sample and split back into the two groups.

    Arguments
    ---------
    - target: AaDataRatio or tuple/list (numerator, denominator)
    - alpha: float - significance level
    - seed: int - seed for random data generation
    - name: str - designer name
    - pval_func: callable - receives 1-D linearized (test, control) arrays
    - effect_sizer: 'percent' | 'const' | 'custom'
    - ratio: 'int' | 'float' - controls numerator rounding when the effect is
      applied (stochastic integer rounding for count metrics).
    """
    def __init__(
        self,
        target: Union[AaDataRatio, Tuple, List],
        alpha: float = 0.05,
        beta: float = 0.20,
        seed: int = 42,
        name: Optional[str] = None,
        pval_func: Optional[Callable] = None,
        effect_sizer: Literal['percent', 'const', 'custom', 'function'] = 'percent',
        ratio: Literal['int', 'float'] = 'int',
        custom_data: Optional[Tuple[NDArray, NDArray]] = None,
        raw_data: Optional[Tuple[NDArray, NDArray]] = None,
        func: Optional[Callable] = None
    ):
        if ratio not in ('int', 'float'):
            raise ValueError("DesignerRatioLin supports ratio in {'int', 'float'}.")

        # If target is not an AaDataRatio, build it under the hood
        if not isinstance(target, AaDataRatio):
            if isinstance(target, (tuple, list)):
                if len(target) != 2:
                    raise ValueError(f"Target tuple/list must have 2 elements (numerator, denominator), got {len(target)}")

                numerator, denominator = target
                numerator = np.array(numerator)
                denominator = np.array(denominator)

                is_proportion = (np.all(numerator >= 0) and
                               np.all(numerator <= denominator) and
                               np.all(numerator == numerator.astype(int)) and
                               np.all(denominator == denominator.astype(int)))

                target = AaDataRatio((numerator, denominator), proportion=is_proportion)
            else:
                raise ValueError("Target must be AaDataRatio or tuple/list of (numerator, denominator)")

        self.target = target
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        self.name = name
        self.pval_func = pval_func
        self.ratio = ratio

        effect_sizer = _EFFECTS_DICT[effect_sizer]
        if effect_sizer == CustomEffectSizer:
            if custom_data is None:
                raise ValueError("custom_data must be provided if effect_sizer is 'custom'")
            if custom_data[0].shape != target.data_raw[0].shape:
                raise ValueError("custom_data must have the same shape as target data")
            custom_data = custom_data
            raw_data = target.data_raw
        elif effect_sizer == FunctionEffectSizer:
            if func is None:
                raise ValueError("func must be provided if effect_sizer is 'function'")

        # The effect is applied to the ratio numerator (with the requested
        # int/float rounding), so the effect sizer is configured as a ratio
        # sizer. The covariate transform is NOT delegated here (covariates=None
        # at call time); it is applied once on the pooled linearized sample.
        self.add_effect = effect_sizer(
            proportion=target.proportion,
            ratio=ratio,
            seed=seed,
            func=func,
            custom_data=custom_data,
            raw_data=raw_data
        )

    def __repr__(self):
        name_str = f"'{self.name}'" if self.name else "None"
        has_pval = self.pval_func is not None
        return f"DesignerRatioLin(name={name_str}, alpha={self.alpha}, has_pval_func={has_pval}, target={self.target})"

    def _single_sim(
        self,
        seed: int,
        pval_func: Callable,
        n_controls: int,
        n_jobs: int,
        idx: np.array,
        effect: float = 0.0,
        return_rejection: bool = False
    ):
        """
        Single AA/AB simulation for a linearized ratio metric.

        The effect is applied to the test numerator, the pooled test+control
        sample is linearized (theta estimated on the control group), and the
        optional transformer is fitted once on the pooled linearized values.
        ``pval_func`` receives 1-D linearized test/control arrays.
        """
        if idx is None:
            idx = self.target.idx.copy()
            if n_jobs == 1:
                self.rng.shuffle(idx)
            else:
                rng = np.random.default_rng(seed)
                rng.shuffle(idx)

        test_idx = idx[:n_controls]
        control_idx = idx[n_controls:]

        num_raw, denom_raw = self.target.data_raw

        # Apply effect only (no covariate transform here), exactly as in the
        # other designers: covariates=None so the transformer is not fitted
        # per-group; it is applied once below on the pooled linearized sample.
        control_num, control_denom = num_raw[control_idx], denom_raw[control_idx]

        if isinstance(self.add_effect, CustomEffectSizer):
            first_argument = test_idx
        elif isinstance(self.add_effect, FunctionEffectSizer):
            first_argument = (
                (num_raw[test_idx], denom_raw[test_idx]),
                (control_num, control_denom),
                (num_raw, denom_raw),
                test_idx, control_idx, idx
            )
        else:
            first_argument = (num_raw[test_idx], denom_raw[test_idx])
        test_num, test_denom = self.add_effect(
            first_argument,
            effect=effect,
            seed=seed
        )

        # Linearization of the ratio metric.
        #
        # Linearizer collapses each group's (numerator, denominator) into a
        # single per-object value num - theta * denom, with theta estimated on
        # the CONTROL group. The result is a plain 1-D metric for each group.
        test, control, _theta = Linearizer(
            (test_num, test_denom),
            (control_num, control_denom)
        )

        # Variance reduction (CUPED / CUPAC / etc.) on the linearized metric.
        #
        # Now that the metric is 1-D this is identical to DesignerIid: the
        # SINGLE transformer is fitted once on the pooled test+control
        # linearized sample (same coefficients for both groups, so the
        # intercept cancels and the type I error stays controlled) and split
        # back. transformer(covariates, lin) -> lin_adj.
        transformer = self.target.transformer
        covariates = self.target.covariates
        if transformer is not None and covariates is not None:
            n_test = len(test)
            pooled_lin = np.concatenate([test, control])
            pooled_cov = _concat_rows(
                _take_rows(covariates, test_idx),
                _take_rows(covariates, control_idx)
            )
            pooled_adjusted = transformer(pooled_cov, pooled_lin)
            test = pooled_adjusted[:n_test]
            control = pooled_adjusted[n_test:]

        pval = pval_func(test, control)

        if return_rejection:
            return int(pval < self.alpha)
        return pval

    def _calculate_some_single_power(
        self,
        seed: int,
        pval_func: Callable,
        n_controls: int,
        effect: float,
        n_jobs: int,
        idx: np.array,
        as_rejection: bool = True
    ):
        """
        Single simulation of power for a linearized ratio metric.

        If ``as_rejection=True`` returns 1 if the test detected an effect at
        self.alpha, 0 otherwise. If ``as_rejection=False`` returns the raw
        p-value instead.

        Delegates to _single_sim with return_rejection=as_rejection.
        """
        return self._single_sim(
            seed=seed,
            pval_func=pval_func,
            n_controls=n_controls,
            n_jobs=n_jobs,
            idx=idx,
            effect=effect,
            return_rejection=as_rejection
        )


class DesignerDept1s(_DesignerBase):
    """
    Designer for A/B tests on dependent one-sample data with an artificial
    (synthetic) control (see :class:`AaDataDept1s`).

    Runs Monte-Carlo AA/AB simulations and power analysis on the per-unit
    difference between the test data and its synthetic control.
    """
    def __init__(
        self,
        target: Union[AaDataDept1s, pd.Series, np.ndarray],
        alpha: float = 0.05,
        beta: float = 0.20,
        seed: int = 42,
        name: Optional[str] = None,
        pval_func: Optional[Callable] = None,
        effect_sizer: Literal['percent', 'const', 'custom', 'function'] = 'percent',
        custom_data: NDArray = None,
        func: Optional[Callable] = None,
        **kwargs
    ):
        # If target is not an AaDataDept1s, build it under the hood
        if not isinstance(target, AaDataDept1s):
            if isinstance(target, tuple):
                target = AaDataDept1s(*target)
            else:
                data = np.array(target)
                target = AaDataDept1s(data)
        
        self.target = target
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        self.name = name
        self.pval_func = pval_func

        effect_sizer = _EFFECTS_DICT[effect_sizer]
        raw_data = None
        if effect_sizer == CustomEffectSizer:
            if custom_data is None:
                raise ValueError("custom_data must be provided if effect_sizer is 'custom'")
            if custom_data.shape != target.data_raw.shape:
                raise ValueError("custom_data must have the same shape as target data")
            custom_data = custom_data
            raw_data = target.data_raw
        elif effect_sizer == FunctionEffectSizer:
            if func is None:
                raise ValueError("func must be provided if effect_sizer is 'function'")

        self.add_effect = effect_sizer(
            proportion=False,
            ratio=False,
            seed=seed,
            func=func,
            custom_data=custom_data,
            raw_data=raw_data
        )
    
    def __repr__(self):
        name_str = f"'{self.name}'" if self.name else "None"
        has_pval = self.pval_func is not None
        return f"DesignerDept(name={name_str}, alpha={self.alpha}, has_pval_func={has_pval}, target={self.target})"

    def _single_sim(
        self,
        seed: int,
        pval_func: Callable,
        n_controls: int,
        n_jobs: int,
        idx: np.array,
        effect: float = 0.0,
        return_rejection: bool = False
    ):
        """
        Single AA/AB simulation.
        
        Parameters
        ----------
        i : int
            Simulation index.
        pval_func : Callable
            Function to calculate p-value.
        n_controls : int
            Number of control observations.
        n_jobs : int
            Number of parallel jobs (affects RNG).
        idx : np.array or None
            Pre-shuffled indices. If None, will shuffle internally.
        effect : float, default=0.0
            Effect size to apply to test group. 0.0 for AA-test.
        return_rejection : bool, default=False
            If True, returns 1/0 for rejection at alpha level. If False, returns p-value.
        
        Returns
        -------
        float or int
            P-value if return_rejection=False, rejection indicator (0/1) if True.
        """
        _data = self.target.data_mod
        n = len(_data)
        if idx is None:
            if n_jobs == 1:
                rng = self.rng
            else:
                rng = np.random.default_rng(seed)
            idx = rng.choice(self.target.idx, size=n, replace=True)

        data = _data[idx] * rng.choice([-1., 1.], size=len(idx))

        # Apply effect only (no covariate transform here).
        #
        # NOTE: unlike DesignerIid/DesignerRatio the effect branch is kept,
        # because this is a ONE-SAMPLE design: the sign-flip above (data =
        # _data[idx] * ±1) is what generates the null distribution for the
        # synthetic control. Unconditionally rebuilding `data` from the raw
        # sample would discard that sign-flip and break the AA simulation.
        # The variance-reduction transform is therefore NOT delegated to
        # add_effect (covariates=None) and is applied once below instead.
        if effect != 0.0:
            if isinstance(self.add_effect, CustomEffectSizer):
                first_argument = idx
            elif isinstance(self.add_effect, FunctionEffectSizer):
                # One-sample design: no control split, so control/control_idx
                # are None and the test sample is the sign-flipped resample.
                first_argument = (data, None, self.target.data_raw, idx, None, idx)
            else:
                first_argument = self.target.data_raw[idx]
            data = self.add_effect(
                first_argument,
                effect=effect,
                seed=seed
            )

        # Variance reduction (CUPED / CUPAC / etc.).
        #
        # This is a single-sample design, so there is no test/control split to
        # pool over: the transformer is fitted once on the whole resampled
        # sample using the matching covariates. Guarded on both transformer and
        # covariates being present so it stays a no-op for the default
        # AaDataDept1s (which carries neither).
        transformer = self.target.transformer
        covariates = self.target.covariates
        if transformer is not None and covariates is not None:
            data = transformer(covariates[idx], data)

        pval = pval_func(data)
        
        if return_rejection:
            return int(pval < self.alpha)
        return pval

    def _calculate_some_single_power(
        self,
        seed: int,
        pval_func: Callable,
        n_controls: int,
        effect: float,
        n_jobs: int,
        idx: np.array,
        as_rejection: bool = True
    ):
        """
        Single simulation of power.

        If ``as_rejection=True`` returns 1 if the test discovered an effect at
        the self.alpha significance level, 0 otherwise. If ``as_rejection=False``
        returns the raw p-value instead.

        Delegates to _single_sim with return_rejection=as_rejection.
        """
        return self._single_sim(
            seed=seed,
            pval_func=pval_func,
            n_controls=n_controls,
            n_jobs=n_jobs,
            idx=idx,
            effect=effect,
            return_rejection=as_rejection
        )
    
class _BenchMarkerBase:
    def validate_pval_funcs(self, pval_func):
        if pval_func is None:
            for designer in self.designers:
                if designer.pval_func is None:
                    raise ValueError('pval_func should be specified in a call or in each designer object.')
    
    def _parse_multi_corrections(self, multi_corrections):
        """
        Parse multi_corrections parameter into list of (name, func) tuples.
        
        Automatically ensures 'no_correction' is always included for baseline comparison.
        
        Parameters
        ----------
        multi_corrections : str, list, or tuple
            Can be:
            - 'all': use all built-in methods
            - str: single built-in method name
            - list of str: multiple built-in method names (can include 'all')
            - tuple (name, func): custom correction method
            - list of tuples: multiple custom methods
            - mixed list: combination of strings, 'all', and tuples
        
        Returns
        -------
        list of tuple
            List of (method_name, correction_function) tuples.
            Always includes 'no_correction' as baseline.
        
        Examples
        --------
        >>> # All built-in methods
        >>> methods = bm._parse_multi_corrections('all')
        
        >>> # All built-in + custom methods
        >>> methods = bm._parse_multi_corrections(['all', ('My Method', my_func)])
        
        >>> # Specific methods + custom
        >>> methods = bm._parse_multi_corrections(['bonferroni', ('Custom', func)])
        """
        result = []
        seen_names = set()
        
        # Helper to add method avoiding duplicates
        def add_method(name, func):
            if name not in seen_names:
                result.append((name, func))
                seen_names.add(name)
        
        # Handle 'all' case
        if multi_corrections == 'all':
            for name in CORRECTION_METHODS.keys():
                add_method(name, lambda pvals, m=name: multiple_pvalue_correction(pvals, method=m))
            return result
        
        # Handle single string
        if isinstance(multi_corrections, str):
            if multi_corrections not in CORRECTION_METHODS:
                raise ValueError(f"Unknown correction method: '{multi_corrections}'. "
                               f"Available methods: {list(CORRECTION_METHODS.keys())}")
            add_method(multi_corrections, lambda pvals, m=multi_corrections: multiple_pvalue_correction(pvals, method=m))
            # Ensure no_correction is present
            if 'no_correction' not in seen_names:
                add_method('no_correction', lambda pvals: multiple_pvalue_correction(pvals, method='no_correction'))
            return result
        
        # Handle single tuple (name, func)
        if isinstance(multi_corrections, tuple) and len(multi_corrections) == 2:
            name, func = multi_corrections
            if not isinstance(name, str):
                raise ValueError(f"Correction method name must be a string, got {type(name)}")
            if not callable(func):
                raise ValueError(f"Correction function must be callable, got {type(func)}")
            add_method(name, func)
            # Ensure no_correction is present
            if 'no_correction' not in seen_names:
                add_method('no_correction', lambda pvals: multiple_pvalue_correction(pvals, method='no_correction'))
            return result
        
        # Handle list
        if isinstance(multi_corrections, (list, tuple)):
            for item in multi_corrections:
                # Special case: 'all' in list
                if item == 'all':
                    for name in CORRECTION_METHODS.keys():
                        add_method(name, lambda pvals, m=name: multiple_pvalue_correction(pvals, method=m))
                # String - built-in method
                elif isinstance(item, str):
                    if item not in CORRECTION_METHODS:
                        raise ValueError(f"Unknown correction method: '{item}'. "
                                       f"Available methods: {list(CORRECTION_METHODS.keys())}")
                    add_method(item, lambda pvals, m=item: multiple_pvalue_correction(pvals, method=m))
                # Tuple (name, func) - custom method
                elif isinstance(item, tuple) and len(item) == 2:
                    name, func = item
                    if not isinstance(name, str):
                        raise ValueError(f"Correction method name must be a string, got {type(name)}")
                    if not callable(func):
                        raise ValueError(f"Correction function must be callable, got {type(func)}")
                    add_method(name, func)
                else:
                    raise ValueError(f"Invalid correction method format: {item}. "
                                   f"Expected 'all', string, or tuple (name, func)")
            
            # Ensure no_correction is always present
            if 'no_correction' not in seen_names:
                # Add at the beginning for baseline comparison
                result.insert(0, ('no_correction', lambda pvals: multiple_pvalue_correction(pvals, method='no_correction')))
            
            return result
        
        raise ValueError(f"Invalid multi_corrections type: {type(multi_corrections)}. "
                       f"Expected 'all', str, tuple, or list")

class BenchMarker(_BenchMarkerBase):
    '''
    Class to compare A/B designers.
    '''
    def __init__(
        self,
        designers: List[DesignerIid],
        alpha: float = 0.05,
        beta: float = 0.20,
        seed: int = 42
    ):
        designers_names   = [d.name for d in designers]
        designers_numbers = list(range(len(designers)))
        labels  = [(name or str(num)) for name, num in zip(designers_names, designers_numbers)]

        if len(labels) != len(set(labels)):
            raise ValueError('Designers must have unique names.')

        # Use `idx` (np.arange(n)) to get the sample size: it works for both
        # iid data (data_raw is a 1-D array) and ratio data (data_raw is a
        # (numerator, denominator) tuple that has no `.shape`).
        required_len = len(designers[0].target.idx)
        for designer in designers:
            if len(designer.target.idx) != required_len:
                raise ValueError('Designers must have the same number of samples.')

        self.designers = designers
        self.labels = labels
        self.alpha = alpha
        self.beta = beta
        self.seed  = seed
    
    def __repr__(self):
        n_designers = len(self.designers)
        labels_str = ', '.join([f"'{label}'" for label in self.labels[:3]])
        if n_designers > 3:
            labels_str += f", ... ({n_designers} total)"
        return f"BenchMarker(n_designers={n_designers}, labels=[{labels_str}], alpha={self.alpha}, beta={self.beta})"

    @staticmethod
    def _effective_variance(designer) -> float:
        """Variance of the metric after applying any variance-reduction transformer.

        Handles all designer types:
        - ``DesignerIid``: 1-D ``data_raw``; if a transformer is present it is
          applied to the *full* dataset (not a split) and the variance of the
          result is returned.
        - ``DesignerRatioLin``: the ratio is linearised (``num - θ·denom`` with
          θ = global ratio) and the optional transformer is then applied.
        - ``DesignerRatio``: per-user ratio ``num/denom``; transformer (if any)
          receives ``(covariates, (num, denom))`` and its output ratio is used.
        """
        target = designer.target

        if isinstance(designer, DesignerIid):
            y = target.data_raw.astype(float)
            if target.transformer is not None and target.covariates is not None:
                y = np.asarray(target.transformer(target.covariates, y), dtype=float)
            return float(np.var(y))

        elif isinstance(designer, DesignerRatioLin):
            num, denom = target.data_raw
            num, denom = num.astype(float), denom.astype(float)
            # Linearise: num - θ·denom with θ estimated on the full dataset
            theta = num.sum() / denom.sum() if denom.sum() != 0 else 0.0
            lin = num - theta * denom
            if target.transformer is not None and target.covariates is not None:
                lin = np.asarray(target.transformer(target.covariates, lin), dtype=float)
            return float(np.var(lin))

        else:  # DesignerRatio
            num, denom = target.data_raw
            num, denom = num.astype(float), denom.astype(float)
            if target.transformer is not None and target.covariates is not None:
                num_adj, denom_adj = target.transformer(target.covariates, (num, denom))
                ratio = np.asarray(num_adj, dtype=float) / np.asarray(denom_adj, dtype=float)
            else:
                ratio = num / denom
            return float(np.var(ratio))

    def compare_variance(self):
        """Compare variance across designers, accounting for any transformer.

        For each designer the *effective* variance is computed: the variance of
        the metric *after* applying the variance-reduction transformer
        (CUPED/CUPAC/PostStratification/…) to the full dataset.  This gives a
        fair apples-to-apples comparison and correctly handles all designer
        types (iid, ratio, linearised ratio).
        """
        designers = self.designers
        competitors_labels = self.labels[1:]

        baseline_value     = self._effective_variance(designers[0])
        competitors_values = [self._effective_variance(designers[i])
                              for i in range(1, len(designers))]

        ax, res = plot_comparison_bars(
            baseline=baseline_value,
            values=competitors_values,
            labels=competitors_labels,
            lesser_is_good=True,
            sort=True,
            ax=None,
            title='Variance comparison',
            ylabel='Variance'
        )

        return ax, res
    
    def aa_benchmark(
        self,
        pval_func: Optional[Callable[[NDArray, NDArray], float]] = None,
        n_sims: int = 5_000,
        control_size: Optional[Union[int, float]] = None,
        n_jobs: int = 1,
        verbose: bool = True,
        plot_hists: bool = True,
        figsize: Optional[tuple] = None
    ) -> Tuple[List[NDArray[np.floating]], List[Dict[str, float]]]:
        """
        AA simulations benchmark to compare Type I error rates across designers.
        
        Runs AA tests (no effect) for each designer and compares:
        - Type I error rate (frequency of false H0 rejections)
        - Mean p-value (should be ~0.5)
        - P-value distribution (should be uniform U[0,1])
        
        Parameters
        ----------
        pval_func : callable, optional
            Function to compute p-value: (test, control) -> float.
            If None, uses pval_func from each designer.
        n_sims : int, default=1_000
            Number of AA simulations.
        control_size : int or float, optional
            Control group size:
            - int > 1: absolute number of observations
            - 0 < float < 1: fraction of total sample size
            - None: half of data (default)
        n_jobs : int, default=1
            Number of parallel processes for simulations.
        verbose : bool, default=True
            Show progress and results.
        plot_hists : bool, default=True
            Show p-value distribution histograms for each designer.
        figsize : tuple, optional
            Figure size (width, height) for the plot.
        
        Returns
        -------
        all_pvalues : list of ndarray
            List of p-value arrays for each designer, shape (n_sims,).
        all_stats : list of dict
            List of statistics dictionaries for each designer:
            - 'mean_p': mean p-value
            - 'mean_p_left', 'mean_p_right': CI for mean
            - 'mean_p_ft': Type I error rate (frequency p < alpha)
            - 'mean_p_ft_left', 'mean_p_ft_right': CI for Type I error
            - 'cummeans': cumulative mean p-values
            - 'ks_statistic', 'ks_p_value': Kolmogorov-Smirnov test for uniformity
        
        Notes
        -----
        Visualization includes 3 main plots:
        1. Horizontal bar chart of Type I error rates with CI
        2. Horizontal bar chart of mean p-values with CI
        3. Convergence plot of cumulative mean p-value
        
        If plot_hists=True, adds p-value distribution histograms.
        
        Color scheme:
        - Green: all checks passed (mean_p ≈ 0.5, FTE ≤ α, KS test OK)
        - Red: serious issues (mean_p far from 0.5 AND FTE > α)
        - Orange: partial issues
        
        Examples
        --------
        >>> from mcab.core import BenchMarker, DesignerIid, AaDataIid, RandomData
        >>> from scipy import stats
        >>>
        >>> # Create designers
        >>> data = RandomData().normal_data()
        >>> designer1 = DesignerIid(AaDataIid(data), name='Normal')
        >>> designer2 = DesignerIid(AaDataIid(data * 1.1), name='Scaled')
        >>>
        >>> # Benchmark
        >>> bm = BenchMarker([designer1, designer2])
        >>> pvals, stats = bm.aa_benchmark(
        ...     pval_func=lambda t, c: stats.ttest_ind(t, c).pvalue,
        ...     n_sims=1000,
        ...     plot_hists=True
        ... )
        """
        self.validate_pval_funcs(pval_func)

        # Collect data for all designers
        all_pvalues = []
        all_stats = []
        
        designers_iter = self.designers
        if verbose:
            designers_iter = tqdm(self.designers, desc='Running AA benchmarks')

        for i, designer in enumerate(designers_iter):
            if verbose:
                print(f'Designer: {self.labels[i]}')
            
            pvalues = designer.aa_sims(
                pval_func=pval_func or designer.pval_func,
                n_sims=n_sims,
                control_size=control_size,
                n_jobs=n_jobs,
                verbose=verbose,
                plot=False
            )

            errors = (pvalues < self.alpha).astype(int)
            cumulative_mean = cum_mean(errors)

            mean_p, mean_p_left, mean_p_right = mean_ci_normal(pvalues, alpha=self.alpha)
            mean_p_ft, mean_p_ft_left, mean_p_ft_right = first_type_error_ci(pvalues, alpha=self.alpha)
            ks_statistic, ks_p_value = stats.kstest(pvalues, 'uniform', args=(0, 1))
            
            all_pvalues.append(pvalues)
            all_stats.append({
                'mean_p': mean_p,
                'mean_p_left': mean_p_left,
                'mean_p_right': mean_p_right,
                'mean_p_ft': mean_p_ft,
                'mean_p_ft_left': mean_p_ft_left,
                'mean_p_ft_right': mean_p_ft_right,
                '_1_type_error_cummeans': cumulative_mean,
                'ks_statistic': ks_statistic,
                'ks_p_value': ks_p_value
            })

        # Visualization
        plot_aa_benchmark(all_stats, self.labels, self.alpha, plot_hists, all_pvalues, figsize)
        
        return all_pvalues, all_stats

    def ab_benchmark(
        self,
        pval_func: Optional[Callable[[NDArray, NDArray], float]] = None,
        bar_effect: Optional[float] = None,
        power_curve_effects: Optional[List[float]] = None,
        effect_sizer: Optional[Literal['percent', 'const', 'custom']] = None,
        n_sims: int = 5_000,
        control_size: Optional[Union[int, float]] = None,
        n_jobs: int = 1,
        verbose: bool = True,
        figsize: Optional[tuple] = None
    ) -> Tuple[List[Dict[str, float]], List[NDArray[np.floating]]]:
        """
        AB simulations benchmark to compare statistical power across designers.
        
        Runs AB tests (with effect) for each designer and compares power metrics.
        Can show both single-effect power bars and full power curves.
        
        Parameters
        ----------
        pval_func : callable, optional
            Function to compute p-value: (test, control) -> float.
            If None, uses pval_func from each designer.
        bar_effect : float, optional
            Single effect size for power bar chart and convergence plot.
            If provided, shows 2 plots: power bars + convergence.
        power_curve_effects : list of float, optional
            List of effect sizes for power curve plot.
            If provided, shows power curves across different effects.
        n_sims : int, default=1_000
            Number of AB simulations per effect size.
        n_controls : int or float, optional
            Control group size:
            - int > 1: absolute number of observations
            - 0 < float < 1: fraction of total sample size
            - None: half of data (default)
        n_jobs : int, default=1
            Number of parallel processes for simulations.
        verbose : bool, default=True
            Show progress and results.
        figsize : tuple, optional
            Figure size (width, height) for the plot.
        
        Returns
        -------
        all_bar_stats : list of dict
            Power statistics for bar_effect (if provided):
            - 'mean_p': mean power
            - 'mean_p_left', 'mean_p_right': CI for power
            - 'cummeans': cumulative mean power
        all_power_curves : list of ndarray
            Power curves for each designer (if power_curve_effects provided),
            shape (len(power_curve_effects),).
        
        Raises
        ------
        ValueError
            If neither bar_effect nor power_curve_effects is specified.
        
        Notes
        -----
        Visualization depends on provided parameters:
        - If bar_effect: shows power bar chart + convergence plot
        - If power_curve_effects: shows power curves
        - If both: shows all 3 plots
        
        All plots include a red dashed line at required_power = 1 - beta.
        
        Examples
        --------
        >>> from mcab.core import BenchMarker, DesignerIid, AaDataIid, RandomData
        >>> from scipy import stats
        >>> import numpy as np
        >>>
        >>> # Create designers
        >>> data = RandomData().normal_data()
        >>> designer1 = DesignerIid(AaDataIid(data), name='Normal')
        >>> designer2 = DesignerIid(AaDataIid(data * 0.9), name='Lower Var')
        >>>
        >>> # Benchmark with single effect
        >>> bm = BenchMarker([designer1, designer2], beta=0.2)
        >>> bar_stats, _ = bm.ab_benchmark(
        ...     pval_func=lambda t, c: stats.ttest_ind(t, c).pvalue,
        ...     bar_effect=0.05,
        ...     n_sims=1000
        ... )
        >>>
        >>> # Benchmark with power curves
        >>> _, curves = bm.ab_benchmark(
        ...     pval_func=lambda t, c: stats.ttest_ind(t, c).pvalue,
        ...     power_curve_effects=np.linspace(0.01, 0.10, 10),
        ...     n_sims=500
        ... )
        """
        self.validate_pval_funcs(pval_func)
        if (not bar_effect) and (not power_curve_effects):
            raise ValueError('bar_effect or power_curve_effects must be specified.')
        
        # Collect data for all designers
        all_bar_stats = []
        all_power_curves = []
        
        _designers_iter = self.designers
        if effect_sizer is None:
            designers_iter = _designers_iter
        else:
            add_effect = _EFFECTS_DICT[effect_sizer]
            designers_iter = deepcopy(_designers_iter)
            for designer in designers_iter:
                designer.add_effect = add_effect
        
        if verbose:
            designers_iter = tqdm(self.designers, desc='Running AB benchmarks')

        for i, designer in enumerate(designers_iter):
            if verbose:
                print(f'Designer: {self.labels[i]}')
            
            bar_stats = None
            if bar_effect:
                bar_rejections = designer.calculate_some_power(
                    effect=bar_effect,
                    pval_func=pval_func or designer.pval_func,
                    control_size=control_size,
                    n_sims=n_sims,
                    n_jobs=n_jobs,
                    verbose=verbose
                )
                mean_p, mean_p_left, mean_p_right = mean_ci_normal(bar_rejections, alpha=self.alpha)
                cummeans = cum_mean(bar_rejections)
                bar_stats = {
                    'mean_p': mean_p,
                    'mean_p_left': mean_p_left,
                    'mean_p_right': mean_p_right,
                    'cummeans': cummeans
                }
                all_bar_stats.append(bar_stats)

            power_curve_data = None
            if power_curve_effects:
                power_curve = designer.calculate_power_curve(
                    effects_list=power_curve_effects,
                    control_size=control_size,
                    n_sims=n_sims,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    plot=False,
                    annotate_curve=False
                )[0]
                power_curve_data = power_curve[1]  # Extract power values
                all_power_curves.append(power_curve_data)

        # Visualization
        plot_ab_benchmark(all_bar_stats, all_power_curves, self.labels, self.beta, bar_effect, power_curve_effects, figsize)
        
        return all_bar_stats, all_power_curves
    
    def directional_analysis(
        self,
        designer1: Union['DesignerIid', 'DesignerRatio'],
        designer2: Union['DesignerIid', 'DesignerRatio'],
        designer1_diff_func: Optional[Callable] = None,
        designer2_diff_func: Optional[Callable] = None,
        n_sims: int = 10_000,
        control_size: Optional[Union[int, float]] = None,
        effect: float = 0.05,
        n_jobs: int = 1,
        verbose: bool = True,
        suptitle: Optional[str] = None,
        figsize: Optional[tuple] = None
    ) -> Dict[str, Union[float, np.ndarray, List[int]]]:
        """
        Directional analysis: compare agreement between two designers.
        
        Compares how consistently two designers (e.g., with and without CUPED)
        estimate treatment effects across the same data splits.
        
        Key metrics:
        - Sign agreement: how often both methods show effect in the same direction
        - R² (explained variance): proportion of variance in designer2 explained by designer1
        - Cloud dispersion: how much the methods diverge from perfect agreement
        
        Parameters
        ----------
        designer1 : DesignerIid or DesignerRatio
            First designer for comparison.
        designer2 : DesignerIid or DesignerRatio
            Second designer for comparison.
        n_sims : int, default=100
            Number of simulations.
        control_size : int or float, optional
            Size of the first (test) group taken from each random split.
            The remaining observations form the control group:
            - int > 1: absolute number of observations
            - 0 < float < 1: fraction of total sample size
            - None: half of data (default)
        effect : float, default=0.05
            Effect size to apply to test group (percentage).
        n_jobs : int, default=1
            Number of parallel processes.
        verbose : bool, default=True
            Show progress bar.
        suptitle : str, optional
            Overall title for plots.
        figsize : tuple, optional
            Figure size (width, height). Default is (20, 16).
        
        Returns
        -------
        results : dict
            Dictionary with analysis results:
            - 'delta1': ndarray of deltas from designer1
            - 'delta2': ndarray of deltas from designer2
            - 'correlation': Pearson correlation between methods
            - 'correlation_pvalue': p-value of the Pearson correlation
            - 'r_squared': R² (proportion of explained variance)
            - 'sign_agreement': proportion of simulations with same sign
            - 'sign_disagreement_rate': proportion with different signs
            - 'disagreement_indices': list of simulation indices with sign disagreement
            - 'rmse': root mean squared error between methods
            - 'mean_absolute_difference': mean absolute difference
        
        Examples
        --------
        >>> from mcab.core import BenchMarker, DesignerIid, AaDataIid, RandomData
        >>>
        >>> # Create two designers
        >>> data = RandomData().normal_data(size=10_000)
        >>> designer1 = DesignerIid(AaDataIid(data), name='Baseline')
        >>> designer2 = DesignerIid(AaDataIid(data * 0.9), name='Reduced Variance')
        >>>
        >>> # Compare methods
        >>> bm = BenchMarker([designer1])
        >>> results = bm.directional_analysis(
        ...     designer1=designer1,
        ...     designer2=designer2,
        ...     n_sims=500,
        ...     effect=0.05
        ... )
        >>>
        >>> print(f"Sign Agreement: {results['sign_agreement']:.3f}")
        >>> print(f"R²: {results['r_squared']:.3f}")
        >>> print(f"Disagreements: {len(results['disagreement_indices'])}")
        
        Notes
        -----
        Both designers must have the same data length. The method uses identical
        random splits for both designers to ensure fair comparison.
        
        The visualization includes:
        1. Scatter plot (delta2 vs delta1) with sign disagreements highlighted
        2. Residuals histogram (delta2 - delta1)
        3. Bland-Altman plot (mean vs difference)
        4. Statistics summary box
        """
        # Validation — use .idx which is set by all data classes (AaDataIid,
        # AaDataRatio, AaDataDept1s) and always has length == n observations.
        len1 = len(designer1.target.idx)
        len2 = len(designer2.target.idx)
        
        if len1 != len2:
            raise ValueError(
                f'Designers must have the same data length. '
                f'Got designer1: {len1}, designer2: {len2}'
            )
        
        n_controls = designer1.validated_n_controls(control_size)
        
        # Parallel simulations with shared indices
        if verbose:
            ParallelClass = ProgressParallel(
                use_tqdm=True,
                total=n_sims,
                desc='Directional analysis',
                unit='sim',
                leave=True,
                n_jobs=n_jobs,
                initializer=_worker_init_warnings
            )
        else:
            ParallelClass = Parallel(n_jobs=n_jobs, initializer=_worker_init_warnings)
        
        if designer1_diff_func is None:
            designer1_diff_func = lambda t, c: np.mean(t) - np.mean(c)
        if designer2_diff_func is None:
            designer2_diff_func = lambda t, c: np.mean(t) - np.mean(c)

        def _apply_iid_transform(target, test, control, test_idx, control_idx):
            """Apply variance-reduction transformer for iid designers (if present)."""
            tr = target.transformer
            cov = target.covariates
            if tr is None or cov is None:
                return test, control
            n_test = len(test)
            pooled = np.concatenate([test, control])
            pooled_cov = _concat_rows(
                _take_rows(cov, test_idx),
                _take_rows(cov, control_idx)
            )
            pooled_adj = tr(pooled_cov, pooled)
            return pooled_adj[:n_test], pooled_adj[n_test:]

        def _apply_ratio_transform(target, test_num, test_denom,
                                   control_num, control_denom,
                                   test_idx, control_idx):
            """Apply variance-reduction transformer for ratio designers (if present)."""
            tr = target.transformer
            cov = target.covariates
            if tr is None or cov is None:
                return test_num, test_denom, control_num, control_denom
            n_test = len(test_num)
            pooled_num   = np.concatenate([test_num,   control_num])
            pooled_denom = np.concatenate([test_denom, control_denom])
            pooled_cov   = _concat_rows(
                _take_rows(cov, test_idx),
                _take_rows(cov, control_idx)
            )
            adj_num, adj_denom = tr(pooled_cov, (pooled_num, pooled_denom))
            return adj_num[:n_test], adj_denom[:n_test], adj_num[n_test:], adj_denom[n_test:]

        def single_sim(sim_idx):
            """Single simulation with shared random split."""
            # Generate shared random indices for this simulation
            rng = np.random.default_rng(sim_idx)
            idx = np.arange(len1)
            rng.shuffle(idx)
            
            test_idx = idx[:n_controls]
            control_idx = idx[n_controls:]

            target1 = designer1.target
            target2 = designer2.target

            # Build control groups. For ratio designers `data_raw` is a
            # (numerator, denominator) tuple, so it must be indexed
            # component-wise rather than with a single fancy-index.
            if isinstance(designer1, DesignerIid):
                control1 = target1.data_raw[control_idx]
            else:
                control1 = (target1.data_raw[0][control_idx], target1.data_raw[1][control_idx])
            if isinstance(designer2, DesignerIid):
                control2 = target2.data_raw[control_idx]
            else:
                control2 = (target2.data_raw[0][control_idx], target2.data_raw[1][control_idx])
            
            # Apply effect (same logic as _single_sim in each designer class)
            if effect != 0.0:
                if isinstance(designer1.add_effect, CustomEffectSizer):
                    first_argument1 = test_idx
                elif isinstance(designer1.add_effect, FunctionEffectSizer):
                    if isinstance(designer1, DesignerIid):
                        test_raw1 = target1.data_raw[test_idx]
                    else:
                        test_raw1 = (target1.data_raw[0][test_idx], target1.data_raw[1][test_idx])
                    first_argument1 = (test_raw1, control1, target1.data_raw, test_idx, control_idx, idx)
                else:
                    if isinstance(designer1, DesignerIid):
                        first_argument1 = target1.data_raw[test_idx]
                    else:
                        first_argument1 = (target1.data_raw[0][test_idx], target1.data_raw[1][test_idx])
                test1 = designer1.add_effect(
                    first_argument1,
                    effect=effect,
                    seed=sim_idx
                )

                if isinstance(designer2.add_effect, CustomEffectSizer):
                    first_argument2 = test_idx
                elif isinstance(designer2.add_effect, FunctionEffectSizer):
                    if isinstance(designer2, DesignerIid):
                        test_raw2 = target2.data_raw[test_idx]
                    else:
                        test_raw2 = (target2.data_raw[0][test_idx], target2.data_raw[1][test_idx])
                    first_argument2 = (test_raw2, control2, target2.data_raw, test_idx, control_idx, idx)
                else:
                    if isinstance(designer2, DesignerIid):
                        first_argument2 = target2.data_raw[test_idx]
                    else:
                        first_argument2 = (target2.data_raw[0][test_idx], target2.data_raw[1][test_idx])
                test2 = designer2.add_effect(
                    first_argument2,
                    effect=effect,
                    seed=sim_idx
                )
            else:
                if isinstance(designer1, DesignerIid):
                    test1 = target1.data_raw[test_idx]
                else:
                    test1 = (target1.data_raw[0][test_idx], target1.data_raw[1][test_idx])
                if isinstance(designer2, DesignerIid):
                    test2 = target2.data_raw[test_idx]
                else:
                    test2 = (target2.data_raw[0][test_idx], target2.data_raw[1][test_idx])

            # Apply variance-reduction transformer (CUPED/CUPAC/etc.) for each
            # designer, mirroring the logic in _single_sim.  The transformer is
            # fitted ONCE on the pooled test+control sample so the SAME
            # coefficients are applied to both groups; the intercept therefore
            # cancels in the comparison and the injected effect is preserved.
            #
            # DesignerRatioLin is special: it must be LINEARIZED first (collapsing
            # (num, denom) -> 1-D), and THEN the iid-style transformer is applied
            # on the linearized values — exactly as in DesignerRatioLin._single_sim.
            # The diff_func therefore receives 1-D linearized arrays, not tuples.
            if isinstance(designer1, DesignerIid):
                test1, control1 = _apply_iid_transform(
                    target1, test1, control1, test_idx, control_idx)
            elif isinstance(designer1, DesignerRatioLin):
                t1_num, t1_denom = test1
                c1_num, c1_denom = control1
                test1, control1, _ = Linearizer(
                    (t1_num, t1_denom), (c1_num, c1_denom))
                test1, control1 = _apply_iid_transform(
                    target1, test1, control1, test_idx, control_idx)
            else:  # DesignerRatio
                t1_num, t1_denom = test1
                c1_num, c1_denom = control1
                t1_num, t1_denom, c1_num, c1_denom = _apply_ratio_transform(
                    target1, t1_num, t1_denom, c1_num, c1_denom, test_idx, control_idx)
                test1, control1 = (t1_num, t1_denom), (c1_num, c1_denom)

            if isinstance(designer2, DesignerIid):
                test2, control2 = _apply_iid_transform(
                    target2, test2, control2, test_idx, control_idx)
            elif isinstance(designer2, DesignerRatioLin):
                t2_num, t2_denom = test2
                c2_num, c2_denom = control2
                test2, control2, _ = Linearizer(
                    (t2_num, t2_denom), (c2_num, c2_denom))
                test2, control2 = _apply_iid_transform(
                    target2, test2, control2, test_idx, control_idx)
            else:  # DesignerRatio
                t2_num, t2_denom = test2
                c2_num, c2_denom = control2
                t2_num, t2_denom, c2_num, c2_denom = _apply_ratio_transform(
                    target2, t2_num, t2_denom, c2_num, c2_denom, test_idx, control_idx)
                test2, control2 = (t2_num, t2_denom), (c2_num, c2_denom)

            delta1 = designer1_diff_func(test1, control1)
            delta2 = designer2_diff_func(test2, control2)
            
            return delta1, delta2
        
        # Run simulations
        results_list = ParallelClass(
            delayed(single_sim)(i) for i in range(n_sims)
        )
        
        # Collect results
        delta1 = np.array([r[0] for r in results_list])
        delta2 = np.array([r[1] for r in results_list])
        
        # Calculate metrics
        # 1. Sign agreement
        sign_agreement = np.mean(np.sign(delta1) == np.sign(delta2))
        sign_disagreement = np.mean(np.sign(delta1) != np.sign(delta2))
        disagreement_indices = np.where(np.sign(delta1) != np.sign(delta2))[0].tolist()
        
        # 2. Pearson correlation and p-value.
        # pearsonr is undefined when either input has zero variance (e.g. all
        # deltas identical for discrete/proportion data); guard against the
        # resulting nan/ConstantInputWarning and report a neutral result.
        if np.std(delta1) == 0 or np.std(delta2) == 0:
            correlation, correlation_pvalue = np.nan, np.nan
        else:
            correlation, correlation_pvalue = stats.pearsonr(delta1, delta2)

        # R^2 (proportion of variance in delta2 explained by delta1).
        r_squared = correlation ** 2 if np.isfinite(correlation) else np.nan

        # 3. Dispersion metrics
        rmse = np.sqrt(np.mean((delta1 - delta2) ** 2))
        mad = np.mean(np.abs(delta1 - delta2))
        
        # Visualization
        plot_directional_analysis(
            delta1=delta1,
            delta2=delta2,
            disagreement_indices=disagreement_indices,
            correlation=correlation,
            correlation_pvalue=correlation_pvalue,
            r_squared=r_squared,
            sign_agreement=sign_agreement,
            sign_disagreement=sign_disagreement,
            rmse=rmse,
            mad=mad,
            suptitle=suptitle,
            figsize=figsize
        )
        
        return {
            'delta1': delta1,
            'delta2': delta2,
            'correlation': correlation,
            'correlation_pvalue': correlation_pvalue,
            'r_squared': r_squared,
            'sign_agreement': sign_agreement,
            'sign_disagreement_rate': sign_disagreement,
            'disagreement_indices': disagreement_indices,
            'rmse': rmse,
            'mean_absolute_difference': mad
        }

    def multi_test_1_type_error(
        self,
        pval_func: Optional[Callable[[NDArray, NDArray], float]] = None,
        multi_corrections: Union[str, List[str]] = 'all',
        n_sims: int = 1_000,
        control_size: Optional[Union[int, float]] = None,
        n_jobs: int = 1,
        verbose: bool = True,
        figsize: Optional[tuple] = None
    ) -> Dict[str, Dict[str, Union[float, List[int]]]]:
        """
        Calculate Family-Wise Error Rate (FWER) for multiple testing corrections.
        
        Runs AA simulations across all designers simultaneously and applies various
        multiple testing correction methods to measure FWER - the probability that
        at least one test produces a false positive when all null hypotheses are true.
        
        Parameters
        ----------
        pval_func : callable, optional
            Function to compute p-value: (test, control) -> float.
            If None, uses pval_func from each designer.
        multi_corrections : str, tuple, or list, default='all'
            Correction methods to compare. Can be:
            - 'all': test all built-in methods
            - str: single built-in method name (e.g., 'bonferroni')
            - list of str: multiple built-in methods (e.g., ['bonferroni', 'holm_bonferroni'])
            - tuple (name, func): custom correction method where:
              * name (str): display name for plots
              * func (callable): correction function that takes array of p-values and returns corrected p-values
            - list of tuples: multiple custom methods
            - mixed list: combination of strings and tuples
            
            Built-in methods: 'no_correction', 'bonferroni', 'holm_bonferroni',
            'hochberg', 'benjamini_hochberg', 'benjamini_yekutieli',
            'benjamini_krieger_yekutieli', 'sidak', 'holm_sidak'
            
            Example with custom method:
            >>> custom_bonf = ('My Bonferroni', lambda pvals: np.minimum(pvals * len(pvals), 1.0))
            >>> results = bm.multi_test_1_type_error(
            ...     pval_func=my_func,
            ...     multi_corrections=['bonferroni', custom_bonf]
            ... )
        n_sims : int, default=1_000
            Number of AA simulations.
        n_controls : int or float, optional
            Control group size (same format as in aa_benchmark).
        n_jobs : int, default=1
            Number of parallel processes (currently sequential for multiple testing).
        verbose : bool, default=True
            Show progress and results.
        figsize : tuple, optional
            Figure size (width, height) for the plot.
        
        Returns
        -------
        results : dict of dict
            Dictionary with correction method names as keys, each containing:
            - 'fwer': observed FWER (proportion of sims with ≥1 rejection)
            - 'fwer_ci_left', 'fwer_ci_right': 95% CI for FWER
            - 'violations': list of binary indicators (1 if any rejection, 0 otherwise)
        
        Notes
        -----
        FWER (Family-Wise Error Rate) is the probability of making at least one
        Type I error (false positive) when testing multiple hypotheses.
        
        Good correction methods should keep FWER ≤ α (typically 0.05).
        
        Visualization includes 2 plots:
        1. Horizontal bar chart of FWER with CI for each method
        2. Convergence plot showing cumulative FWER over simulations
        
        Color scheme:
        - Green: FWER ≤ α (good control)
        - Red: FWER > α (poor control)
        
        Examples
        --------
        >>> from mcab.core import BenchMarker, DesignerIid, AaDataIid, RandomData
        >>> from scipy import stats
        >>>
        >>> # Create multiple designers
        >>> data = RandomData().normal_data()
        >>> designers = [
        ...     DesignerIid(AaDataIid(data), name=f'Designer{i}')
        ...     for i in range(5)
        ... ]
        >>>
        >>> # Test FWER control
        >>> bm = BenchMarker(designers, alpha=0.05)
        >>> results = bm.multi_test_1_type_error(
        ...     pval_func=lambda t, c: stats.ttest_ind(t, c).pvalue,
        ...     multi_corrections=['no_correction', 'bonferroni', 'benjamini_hochberg'],
        ...     n_sims=1000
        ... )
        >>>
        >>> # Check which methods control FWER
        >>> for method, res in results.items():
        ...     print(f"{method}: FWER = {res['fwer']:.3f}")
        """
        self.validate_pval_funcs(pval_func)
        
        # Parse correction methods (supports strings, tuples with custom functions, and lists)
        correction_methods = self._parse_multi_corrections(multi_corrections)
        
        n_controls = self.designers[0].validated_n_controls(control_size)
        n_designers = len(self.designers)
        
        # Run simulations with parallelization
        if verbose:
            print(f'Running {n_sims} AA simulations for {n_designers} designers...')
        
        if verbose and n_jobs != 1:
            ParallelClass = ProgressParallel(
                use_tqdm=True,
                total=n_sims,
                desc='AA simulations',
                unit='sim',
                leave=True,
                n_jobs=n_jobs,
                initializer=_worker_init_warnings
            )
        else:
            ParallelClass = Parallel(n_jobs=n_jobs, initializer=_worker_init_warnings)
        
        def single_aa_sim(sim_idx):
            # Generate shared random indices for this simulation
            rng = np.random.default_rng(sim_idx)
            idx = np.arange(len(self.designers[0].target.data_raw))
            rng.shuffle(idx)
            
            pvalues_sim = []
            for designer in self.designers:
                pval = designer._single_sim(sim_idx, pval_func or designer.pval_func, n_controls, n_jobs, idx, effect=0.0, return_rejection=False)
                pvalues_sim.append(pval)
            
            return pvalues_sim
        
        if n_jobs == 1:
            all_pvalues_matrix = []
            for sim_idx in tqdm(range(n_sims), disable=not verbose, desc='AA simulations'):
                all_pvalues_matrix.append(single_aa_sim(sim_idx))
        else:
            all_pvalues_matrix = ParallelClass(
                delayed(single_aa_sim)(sim_idx) for sim_idx in range(n_sims)
            )
        
        all_pvalues_matrix = np.array(all_pvalues_matrix)  # shape: (n_sims, n_designers)
        
        # Calculate FWER for each correction method
        results = {}
        
        for method_name, correction_func in correction_methods:
            fwer_violations = []
            
            for sim_idx in range(n_sims):
                pvalues = all_pvalues_matrix[sim_idx, :]
                
                # Apply correction using the provided function
                adjusted_pvalues = correction_func(pvalues)
                
                # Check if at least one test rejects H0 (false positive)
                any_rejection = np.any(adjusted_pvalues < self.alpha)
                fwer_violations.append(int(any_rejection))
            
            fwer = np.mean(fwer_violations)
            fwer_ci_left, fwer_ci_right = (
                fwer - 1.96 * np.sqrt(fwer * (1 - fwer) / n_sims),
                fwer + 1.96 * np.sqrt(fwer * (1 - fwer) / n_sims)
            )
            
            results[method_name] = {
                'fwer': fwer,
                'fwer_ci_left': max(0, fwer_ci_left),
                'fwer_ci_right': min(1, fwer_ci_right),
                'violations': fwer_violations
            }
        
        # Visualization
        plot_multi_test_fwer(results, self.alpha, n_designers, figsize)
        
        return results

    def multi_test_power(
        self,
        pval_func: Optional[Callable[[NDArray, NDArray], float]] = None,
        bar_effect: Optional[Union[float, List[float]]] = None,
        multi_corrections: Union[str, List[str]] = 'all',
        power_curve_effects: Optional[List[float]] = None,
        n_sims: int = 1_000,
        control_size: Optional[Union[int, float]] = None,
        n_jobs: int = 1,
        verbose: bool = True,
        figsize: Optional[tuple] = None
    ) -> Dict[str, Dict[str, Union[float, NDArray[np.int_]]]]:
        """
        Calculate power metrics for multiple testing with various correction methods.
        
        Runs AB simulations across all designers simultaneously with specified effects
        and applies multiple testing corrections to measure different power metrics.
        Optionally sweeps over a range of effect sizes to build power curves.
        
        Parameters
        ----------
        pval_func : callable, optional
            Function to compute p-value: (test, control) -> float.
            If None, uses pval_func from each designer.
        bar_effect : float or list of float, optional
            Effect size(s) applied to all designers for the bar-chart summary.
            - float: same effect for all designers
            - list of float: one effect per designer (length must equal number of designers)
            - None: defaults to 0.05 for all designers
        multi_corrections : str, tuple, or list, default='all'
            Correction methods to compare. Same format as in multi_test_1_type_error:
            - 'all': test all built-in methods
            - str: single built-in method name
            - list of str: multiple built-in methods
            - tuple (name, func): custom correction method
            - list of tuples: multiple custom methods
            - mixed list: combination of strings and tuples
            
            See multi_test_1_type_error documentation for detailed examples.
        power_curve_effects : list of float, optional
            Sequence of uniform effect sizes to sweep over for power curves.
            At each point the same scalar effect is applied to **all** designers.
            When provided, a second row of plots (one curve per correction method)
            is added to the visualization, and each entry in ``results`` gains a
            ``'power_curve'`` sub-dict with keys ``'avg_power'``,
            ``'any_pair_power'`` and ``'all_pairs_power'`` (lists aligned with
            ``power_curve_effects``).
        n_sims : int, default=1_000
            Number of AB simulations per effect size point.
        control_size : int or float, optional
            Control group size (same format as in aa_benchmark).
        n_jobs : int, default=1
            Number of parallel processes.
        verbose : bool, default=True
            Show progress and results.
        figsize : tuple, optional
            Figure size (width, height) for the plot.
        
        Returns
        -------
        results : dict of dict
            Dictionary with correction method names as keys, each containing:
            - 'avg_power': average power (mean rejections / total tests)
            - 'any_pair_power': probability of detecting ≥1 effect
            - 'all_pairs_power': probability of detecting all effects
            - 'rejections': array of rejection counts per simulation
            - 'power_curve': dict with 'avg_power', 'any_pair_power',
              'all_pairs_power' lists (only when ``power_curve_effects`` is given)
        
        Notes
        -----
        Power metrics explained:
        
        - **Average Power** (AvgP): Mean proportion of true effects detected.
          Formula: E[detected effects] / total effects
          
        - **Any-Pair Power** (APP): Probability of detecting at least one effect.
          Formula: P(rejections ≥ 1)
          
        - **All-Pairs Power** (AllP): Probability of detecting all effects.
          Formula: P(rejections = total tests)
        
        Visualization includes 3 bar-chart plots:
        1. Average Power comparison across methods
        2. Any-Pair Power comparison
        3. All-Pairs Power comparison
        
        When ``power_curve_effects`` is supplied, a second row of 3 line plots
        (one curve per correction method) is added below.
        
        Trade-off: Methods with stricter error control (e.g., Bonferroni) typically
        have lower power than methods with relaxed control (e.g., Benjamini-Hochberg).
        
        Examples
        --------
        >>> from mcab.core import BenchMarker, DesignerIid, AaDataIid, RandomData
        >>> from scipy import stats
        >>> import numpy as np
        >>>
        >>> # Create multiple designers
        >>> data = RandomData().normal_data()
        >>> designers = [
        ...     DesignerIid(AaDataIid(data), name=f'Designer{i}')
        ...     for i in range(5)
        ... ]
        >>>
        >>> # Test power with same effect for all
        >>> bm = BenchMarker(designers, alpha=0.05)
        >>> results = bm.multi_test_power(
        ...     pval_func=lambda t, c: stats.ttest_ind(t, c).pvalue,
        ...     bar_effect=0.05,
        ...     multi_corrections=['no_correction', 'bonferroni', 'benjamini_hochberg'],
        ...     n_sims=1000
        ... )
        >>>
        >>> # Power curves across a range of effects
        >>> results = bm.multi_test_power(
        ...     pval_func=lambda t, c: stats.ttest_ind(t, c).pvalue,
        ...     bar_effect=0.05,
        ...     power_curve_effects=np.linspace(0.01, 0.10, 10),
        ...     n_sims=500
        ... )
        >>>
        >>> # Different effects per designer
        >>> results2 = bm.multi_test_power(
        ...     pval_func=lambda t, c: stats.ttest_ind(t, c).pvalue,
        ...     bar_effect=[0.02, 0.03, 0.05, 0.07, 0.10],
        ...     n_sims=500
        ... )
        """
        self.validate_pval_funcs(pval_func)
        
        # Parse correction methods (supports strings, tuples with custom functions, and lists)
        correction_methods = self._parse_multi_corrections(multi_corrections)
        
        # Normalise bar_effect to a per-designer list
        if bar_effect is None:
            bar_effect = [0.05] * len(self.designers)
        elif isinstance(bar_effect, (int, float)):
            bar_effect = [bar_effect] * len(self.designers)
        
        n_controls = self.designers[0].validated_n_controls(control_size)
        n_designers = len(self.designers)
        
        # Run simulations with parallelization
        if verbose:
            print(f'Running {n_sims} AB simulations for {n_designers} designers with bar_effect {bar_effect}...')
        
        if verbose and n_jobs != 1:
            ParallelClass = ProgressParallel(
                use_tqdm=True,
                total=n_sims,
                desc='AB simulations',
                unit='sim',
                leave=True,
                n_jobs=n_jobs,
                initializer=_worker_init_warnings
            )
        else:
            ParallelClass = Parallel(n_jobs=n_jobs, initializer=_worker_init_warnings)
        
        def _run_ab_sims(effects_list):
            """Run n_sims AB simulations and return (n_sims, n_designers) p-value matrix."""
            def single_ab_sim(sim_idx):
                rng = np.random.default_rng(sim_idx)
                idx = np.arange(len(self.designers[0].target.data_raw))
                rng.shuffle(idx)
                return [
                    designer._single_sim(
                        sim_idx, pval_func or designer.pval_func,
                        n_controls, n_jobs, idx,
                        effect=eff, return_rejection=False
                    )
                    for designer, eff in zip(self.designers, effects_list)
                ]

            if n_jobs == 1:
                return np.array([single_ab_sim(i) for i in range(n_sims)])
            return np.array(ParallelClass(delayed(single_ab_sim)(i) for i in range(n_sims)))

        if n_jobs == 1:
            all_pvalues_matrix = []
            for sim_idx in tqdm(range(n_sims), disable=not verbose, desc='AB simulations'):
                rng = np.random.default_rng(sim_idx)
                idx = np.arange(len(self.designers[0].target.data_raw))
                rng.shuffle(idx)
                row = [
                    designer._single_sim(
                        sim_idx, pval_func or designer.pval_func,
                        n_controls, n_jobs, idx,
                        effect=eff, return_rejection=False
                    )
                    for designer, eff in zip(self.designers, bar_effect)
                ]
                all_pvalues_matrix.append(row)
        else:
            all_pvalues_matrix = ParallelClass(
                delayed(lambda i: [
                    designer._single_sim(
                        i, pval_func or designer.pval_func,
                        n_controls, n_jobs,
                        np.random.default_rng(i).permutation(len(self.designers[0].target.data_raw)),
                        effect=eff, return_rejection=False
                    )
                    for designer, eff in zip(self.designers, bar_effect)
                ])(sim_idx) for sim_idx in range(n_sims)
            )
        
        all_pvalues_matrix = np.array(all_pvalues_matrix)  # shape: (n_sims, n_designers)
        
        def _compute_metrics(pv_matrix):
            """Compute avg/any/all power per correction method from a p-value matrix."""
            out = {}
            for method_name, correction_func in correction_methods:
                rejs = np.array([
                    np.sum(correction_func(pv_matrix[i]) < self.alpha)
                    for i in range(pv_matrix.shape[0])
                ])
                out[method_name] = {
                    'avg_power':       float(np.mean(rejs) / n_designers),
                    'any_pair_power':  float(np.mean(rejs > 0)),
                    'all_pairs_power': float(np.mean(rejs == n_designers)),
                    'rejections':      rejs,
                }
            return out

        results = _compute_metrics(all_pvalues_matrix)
        
        # ── Power curves ──────────────────────────────────────────────────────
        if power_curve_effects is not None:
            curve_iter = (
                tqdm(power_curve_effects, desc='Power curves', leave=True)
                if verbose else power_curve_effects
            )
            # Initialise per-method curve containers
            for method_name, _ in correction_methods:
                results[method_name]['power_curve'] = {
                    'avg_power': [], 'any_pair_power': [], 'all_pairs_power': []
                }

            for curve_eff in curve_iter:
                curve_effects_list = [curve_eff] * n_designers
                curve_pv_matrix = _run_ab_sims(curve_effects_list)
                curve_metrics = _compute_metrics(curve_pv_matrix)
                for method_name, _ in correction_methods:
                    cm = curve_metrics[method_name]
                    pc = results[method_name]['power_curve']
                    pc['avg_power'].append(cm['avg_power'])
                    pc['any_pair_power'].append(cm['any_pair_power'])
                    pc['all_pairs_power'].append(cm['all_pairs_power'])
        
        # Visualization
        plot_multi_test_power(
            results, n_designers,
            power_curve_effects=power_curve_effects,
            beta=self.beta,
            figsize=figsize,
        )
        
        return results

if __name__ == '__main__':
    from .tests import run_exact_tests
    run_exact_tests()
