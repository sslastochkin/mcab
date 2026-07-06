import os, warnings
import numpy as np
from scipy import stats

from typing import Dict, Optional, Tuple, Union, Callable, Any
from numpy.typing import ArrayLike, NDArray

from enum import IntEnum

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from .utils import ProgressParallel, _worker_init_warnings

# ------------------------------------------------------------------------------------------------ #
#                                        Utility Functions                                         #
# ------------------------------------------------------------------------------------------------ #

__all__ = [
    # Enums
    'Alternative',
    # Utilities
    'apply_statistic',
    # Bootstrap tests
    'bootstrap_1s',
    'bootstrap_2s',
    # Permutation tests
    'sign_permutation_test',
    'permutation_test_2s',
    # Bucket test
    'bucket_test_2s',
    # delta method
    'delta_method_pvalue',
    # Multiple testing corrections
    'multiple_pvalue_correction',
    'CORRECTION_METHODS',
    'bonferroni',
    'holm_bonferroni',
    'hochberg',
    'benjamini_hochberg',
    'benjamini_yekutieli',
    'benjamini_krieger_yekutieli',
    'sidak',
    'holm_sidak',
    'no_multiple_correction'
]

class Alternative(IntEnum):
    TWO_SIDED = 0
    GREATER   = 1
    LESS      = 2
    INTERVAL  = 3

_ALTERNATIVE_MAP: dict[str, Alternative] = {
    'two-sided': Alternative.TWO_SIDED,
    'greater':   Alternative.GREATER,
    'less':      Alternative.LESS,
    'interval':  Alternative.INTERVAL,
}

_ALTERNATIVE_FOR_PVALUE = [Alternative.TWO_SIDED, Alternative.GREATER, Alternative.LESS]

def _parse_alternative(
    value: str | Alternative,
    needed_for_pvalue: bool = False
) -> Alternative:
    if isinstance(value, Alternative):
        res = value
    else:
        try:
            res = _ALTERNATIVE_MAP[value]
        except KeyError:
            raise ValueError(
                f"alternative must be one of {list(_ALTERNATIVE_MAP)}, got '{value}'"
            )
    if needed_for_pvalue and res not in _ALTERNATIVE_FOR_PVALUE:
        raise ValueError(f"alternative must be 'two-sided', 'greater' or 'less', got '{value}'")
    return res


def _resolve_n_jobs(n_jobs: int) -> int:
    '''
    Resolve n_jobs parameter with nested parallelism detection.
    
    Detects if we're already inside a parallel worker and disables
    inner parallelism to avoid process explosion.
    
    Parameters
    ----------
    n_jobs : int
        Requested number of parallel jobs:
        - 1: no parallelism (always safe)
        - -1: use all available CPUs
        - n > 1: use n CPUs
    
    Returns
    -------
    int
        Actual number of jobs to use (1 if nested, otherwise n_jobs).
    '''
    if n_jobs == 1:
        return 1
    
    # Detect nested parallelism via joblib/loky environment variable
    if 'LOKY_MAX_CPU_COUNT' in os.environ:
        return 1
    
    # Convert -1 to actual CPU count
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    
    return max(1, n_jobs)


def apply_statistic(
    samples: ArrayLike,
    statistic: Callable,
    statistic_args: Optional[Tuple] = None,
    statistic_kwargs: Optional[Dict[str, Any]] = None,
    axis: Optional[int] = None
) -> Union[float, NDArray[np.floating]]:
    '''
    Apply a statistic function to samples with optional vectorization.
    
    This helper function attempts to apply a statistic along a specified axis.
    If the statistic doesn't support the axis parameter, it falls back to
    iterating over groups.
    
    Parameters
    ----------
    samples : array-like
        Input data.
    statistic : callable
        Function to compute (e.g., np.mean, np.median).
    statistic_args : tuple, optional
        Additional positional arguments for statistic.
    statistic_kwargs : dict, optional
        Additional keyword arguments for statistic.
    axis : {0, 1, None}, optional
        Axis along which to apply the statistic:
        - None: apply to entire array
        - 0: apply along columns (across rows)
        - 1: apply along rows (across columns)
    
    Returns
    -------
    scalar or ndarray
        Result of applying statistic to samples.
    '''
    assert axis in (0, 1, None)

    statistic_args   = statistic_args or ()
    statistic_kwargs = statistic_kwargs or {}

    samples = np.asarray(samples)

    if axis is None:
        return statistic(samples, *statistic_args, **statistic_kwargs)
        
    try:
        # Try vectorized computation with axis parameter
        return statistic(
            samples,
            *statistic_args,
            **statistic_kwargs,
            axis=axis
        )
    except TypeError:
        # Fallback: iterate over groups if statistic doesn't support axis
        return np.array([
            statistic(group, *statistic_args, **statistic_kwargs)
            for group in (samples if axis == 1 else samples.T)
        ])

# ------------------------------------------------------------------------------------------------ #
#                                        Statistical Tests                                         #
# ------------------------------------------------------------------------------------------------ #

def _boot_pval(
    boot_stats: ArrayLike,
    loc: float = 0,
    alternative: Union[str, Alternative] = 'two-sided'
) -> float:
    '''
    Compute bootstrap p-value.
    
    Tests H0: bootstrap distribution is symmetric around loc.
    
    Parameters
    ----------
    boot_stats : array-like
        Bootstrap statistics distribution.
    loc : float, default=0
        Null hypothesis location parameter.
    alternative : {'two-sided', 'greater', 'less'}
        Alternative hypothesis:
        - 'two-sided': H1: true value != loc
        - 'greater':   H1: true value > loc
        - 'less':      H1: true value < loc
    
    Returns
    -------
    float
        P-value for the test.
    '''
    alt = _parse_alternative(alternative, needed_for_pvalue=True)

    B = len(boot_stats)
    k = np.sum(boot_stats <= loc)   # number of statistics <= loc

    if alt == Alternative.TWO_SIDED:
        # Two-sided: use reflection method
        p_value = 2 * min(
            (k + 1) / (B + 1),
            1 - (k + 1) / (B + 1)
        )
    elif alt == Alternative.GREATER:
        # H1: statistic > loc -> p-value = P(boot_stats <= loc)
        p_value = (k + 1) / (B + 1)
    elif alt == Alternative.LESS:
        # H1: statistic < loc -> p-value = P(boot_stats > loc)
        p_value = 1 - (k + 1) / (B + 1)
    else:
        raise ValueError(f"alternative must be 'two-sided', 'greater' or 'less', got '{alternative}'")

    return min(p_value, 1.0)

def _perm_pval(
    t_perm: NDArray[np.floating],
    t_obs: float,
    alternative: Union[Alternative, str] = 'two-sided'
) -> float:
    '''
    Compute permutation test p-value.
    
    Unified p-value calculation for permutation tests based on the
    null distribution and observed statistic.
    
    Parameters
    ----------
    t_perm : array-like
        Permutation distribution of the test statistic.
    t_obs : float
        Observed test statistic.
    alternative : Alternative
        Alternative hypothesis (must be already parsed to Alternative enum).
    
    Returns
    -------
    float
        P-value for the test.
    
    Notes
    -----
    P-value formulas:
    - TWO_SIDED: P(|t_perm| >= |t_obs|)
    - GREATER:   P(t_perm >= t_obs)
    - LESS:      P(t_perm <= t_obs)
    '''
    alt = _parse_alternative(alternative, needed_for_pvalue=True)
    if alt == Alternative.TWO_SIDED:
        # Two-sided: compare absolute values
        pval = np.mean(np.abs(t_perm) >= np.abs(t_obs))
    elif alt == Alternative.GREATER:
        # One-sided greater: t_perm >= t_obs
        pval = np.mean(t_perm >= t_obs)
    else:  # Alternative.LESS
        # One-sided less: t_perm <= t_obs
        pval = np.mean(t_perm <= t_obs)
    
    return pval

def _one_sample_bootstrap_chunk(
    x_data: NDArray[np.floating],
    n_iterations: int,
    seed: int,
    statistic: Callable,
    statistic_args: Optional[Tuple],
    statistic_kwargs: Optional[Dict[str, Any]]
) -> NDArray[np.floating]:
    '''
    Worker function for parallel one-sample bootstrap.
    
    Generates a chunk of bootstrap resamples and computes statistics.
    Thread limiting is applied to avoid NumPy BLAS oversubscription.
    
    Parameters
    ----------
    x_data : ndarray
        Original sample data.
    n_iterations : int
        Number of bootstrap iterations for this chunk.
    seed : int
        Random seed for this worker.
    statistic : callable
        Statistic function to apply.
    statistic_args : tuple or None
        Additional positional arguments for statistic.
    statistic_kwargs : dict or None
        Additional keyword arguments for statistic.
    
    Returns
    -------
    ndarray
        Bootstrap statistics for this chunk.
    '''
    with threadpool_limits(limits=1, user_api='blas'):
        rng = np.random.default_rng(seed)
        n = len(x_data)
        
        # Generate bootstrap samples (resample with replacement)
        idx = rng.integers(0, n, size=(n_iterations, n))
        samples = x_data[idx]
        
        # Compute statistic for each bootstrap sample
        boot_stats = apply_statistic(
            samples,
            statistic,
            statistic_args=statistic_args,
            statistic_kwargs=statistic_kwargs,
            axis=1
        )
        
        return boot_stats

def bootstrap_1s(
    x: ArrayLike,
    statistic: Callable = np.mean,
    n_bootstrap: int = 5_000,
    statistic_args: Optional[Tuple] = None,
    statistic_kwargs: Optional[Dict[str, Any]] = None,
    rng: Optional[Union[int, np.random.Generator]] = None,
    alternative: Union[str, Alternative] = 'two-sided',
    n_jobs: int = 1,
    verbose: bool = False
) -> Union[float, NDArray[np.floating]]:
    '''
    One-sample bootstrap test.
    
    Performs bootstrap resampling to test H0: statistic(x) = 0.
    
    Parameters
    ----------
    x : array-like
        Input sample.
    statistic : callable, default=np.mean
        Statistic to compute on each bootstrap sample.
    n_bootstrap : int, default=5_000
        Number of bootstrap resamples.
    statistic_args : tuple, optional
        Additional positional arguments for statistic.
    statistic_kwargs : dict, optional
        Additional keyword arguments for statistic.
    rng : int or np.random.Generator, optional
        Random number generator or seed. If None, uses default_rng(42).
    alternative : {'two-sided', 'greater', 'less', 'interval'}, default='two-sided'
        Alternative hypothesis:
        - 'two-sided': H1: statistic(x) != 0
        - 'greater':   H1: statistic(x) > 0
        - 'less':      H1: statistic(x) < 0
        - 'interval':  return bootstrap distribution for CI construction
    n_jobs : int, default=1
        Number of parallel jobs:
        - 1: no parallelism (safe for nested calls, default)
        - -1: use all available CPUs
        - n > 1: use n CPUs
        
        Performance recommendations based on benchmarks:
        - n_bootstrap <= 1000: always use n_jobs=1 (overhead too high)
        - n_bootstrap >= 5000 and sample_size >= 1000: use n_jobs=-1 (3-6x speedup)
        - Very large samples (>20k): overhead may dominate, test both
        
        Automatically detects nested parallelism (e.g., when called from
        DesignerIid.aa_sims) and disables inner parallelism to avoid
        process explosion.
    verbose : bool, default=False
        If True, show progress bar for parallel execution using tqdm.
        Only has effect when n_jobs > 1.
    
    Returns
    -------
    float or ndarray
        If alternative != 'interval': p-value
        If alternative == 'interval': bootstrap distribution
    '''
    alt = _parse_alternative(alternative)

    x = np.asarray(x)
    n = len(x)
    
    # Initialize RNG
    if rng is None:
        rng = np.random.default_rng(42)
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)
    
    # Resolve n_jobs with nested parallelism detection
    n_jobs_actual = _resolve_n_jobs(n_jobs)
    
    # Sequential execution
    if n_jobs_actual == 1:
        # Generate bootstrap samples (resample with replacement)
        idx = rng.integers(0, n, size=(n_bootstrap, n))
        samples = x[idx]

        # Compute statistic for each bootstrap sample
        boot_stats = apply_statistic(
            samples,
            statistic,
            statistic_args=statistic_args,
            statistic_kwargs=statistic_kwargs,
            axis=1
        )
    else:
        # Parallel execution: split n_bootstrap into chunks
        chunk_size = n_bootstrap // n_jobs_actual
        remainder = n_bootstrap % n_jobs_actual
        
        # Generate unique seeds for each worker
        seeds = rng.integers(0, 2**31, size=n_jobs_actual)
        chunk_sizes = [chunk_size + (1 if i < remainder else 0) for i in range(n_jobs_actual)]
        
        if verbose:
            ParallelClass = ProgressParallel(
                use_tqdm=True,
                total=n_jobs_actual,
                desc='Bootstrap 1s chunks',
                unit='chunk',
                leave=True,
                n_jobs=n_jobs_actual,
                backend='loky',
                initializer=_worker_init_warnings
            )
        else:
            ParallelClass = Parallel(n_jobs=n_jobs_actual, backend='loky', initializer=_worker_init_warnings)

        # Compute chunks in parallel (thread limiting is inside chunk function)
        chunks = ParallelClass(
            delayed(_one_sample_bootstrap_chunk)(
                x, chunk_sizes[i], seeds[i],
                statistic, statistic_args, statistic_kwargs
            )
            for i in range(n_jobs_actual)
        )
        
        # Concatenate results from all workers
        boot_stats = np.concatenate(chunks)

    # Return distribution for interval construction
    if alt == Alternative.INTERVAL:
        return boot_stats

    # Compute p-value
    return _boot_pval(boot_stats, alternative=alt)


def _two_sample_test_chunk(
    test_data: Optional[NDArray[np.floating]],
    control_data: Optional[NDArray[np.floating]],
    pooled_data: Optional[NDArray[np.floating]],
    n_iterations: int,
    seed: int,
    statistic: Callable,
    statistic_args: Optional[Tuple],
    statistic_kwargs: Optional[Dict[str, Any]],
    test_type: str  # 'bootstrap' or 'permutation'
) -> NDArray[np.floating]:
    '''
    Unified helper function for parallel two-sample test computation.
    
    Computes a chunk of test statistics for either bootstrap or permutation tests.
    Uses threadpool_limits to prevent BLAS oversubscription in parallel workers.
    
    Parameters
    ----------
    test_data : ndarray or None
        Test group observations (for bootstrap) or None (for permutation).
    control_data : ndarray or None
        Control group observations (for bootstrap) or None (for permutation).
    pooled_data : ndarray or None
        Pooled observations (for permutation) or None (for bootstrap).
    n_iterations : int
        Number of iterations (bootstrap resamples or permutations) in this chunk.
    seed : int
        Random seed for this chunk.
    statistic : callable
        Statistic function to apply.
    statistic_args : tuple or None
        Additional positional arguments for statistic.
    statistic_kwargs : dict or None
        Additional keyword arguments for statistic.
    test_type : {'bootstrap', 'permutation'}
        Type of test to perform.
    
    Returns
    -------
    ndarray
        Array of test statistics (differences) for this chunk.
    
    Notes
    -----
    This function uses threadpool_limits to set BLAS threads to 1, preventing
    oversubscription when multiple workers are running in parallel.
    '''
    # Limit NumPy BLAS threads to avoid oversubscription in parallel workers
    with threadpool_limits(limits=1, user_api='blas'):
        # Initialize RNG with unique seed for this chunk
        rng = np.random.default_rng(seed)
        
        if test_type == 'bootstrap':
            # Bootstrap: resample with replacement from original groups
            b_test = rng.choice(test_data, size=(n_iterations, len(test_data)), replace=True)
            b_control = rng.choice(control_data, size=(n_iterations, len(control_data)), replace=True)
            
            # Compute statistics
            stat_test = apply_statistic(
                samples=b_test,
                statistic=statistic,
                statistic_args=statistic_args,
                statistic_kwargs=statistic_kwargs,
                axis=1
            )
            stat_control = apply_statistic(
                samples=b_control,
                statistic=statistic,
                statistic_args=statistic_args,
                statistic_kwargs=statistic_kwargs,
                axis=1
            )
            
        elif test_type == 'permutation':
            # Permutation: randomly reassign group labels from pooled data
            n_test = len(test_data) if test_data is not None else 0
            n_control = len(control_data) if control_data is not None else 0
            n_total = n_test + n_control
            
            # Generate permutations: randomly assign group labels (vectorized)
            perm_matrix = np.argsort(rng.random((n_iterations, n_total)), axis=1)
            
            # Split into test and control groups for all permutations
            perm_test = pooled_data[perm_matrix[:, :n_test]]
            perm_control = pooled_data[perm_matrix[:, n_test:]]
            
            # Compute statistics
            stat_test = apply_statistic(
                samples=perm_test,
                statistic=statistic,
                statistic_args=statistic_args,
                statistic_kwargs=statistic_kwargs,
                axis=1
            )
            stat_control = apply_statistic(
                samples=perm_control,
                statistic=statistic,
                statistic_args=statistic_args,
                statistic_kwargs=statistic_kwargs,
                axis=1
            )
        else:
            raise ValueError(f"test_type must be 'bootstrap' or 'permutation', got '{test_type}'")
        
        # Return differences
        return stat_test - stat_control


def bootstrap_2s(
    test: ArrayLike,
    control: ArrayLike,
    statistic: Callable = np.mean,
    statistic_args: Optional[Tuple] = None,
    statistic_kwargs: Optional[Dict[str, Any]] = None,
    n_bootstrap: int = 5_000,
    rng: Optional[Union[int, np.random.Generator]] = None,
    alternative: Union[str, Alternative] = 'two-sided',
    n_jobs: int = 1,
    verbose: bool = False
) -> Union[float, NDArray[np.floating]]:
    '''
    Two-sample bootstrap test.
    
    Performs independent bootstrap resampling on two samples to test
    H0: statistic(test) = statistic(control).
    
    Parameters
    ----------
    test : array-like
        Test group observations.
    control : array-like
        Control group observations.
    statistic : callable, default=np.mean
        Statistic to compute on each group.
    statistic_args : tuple, optional
        Additional positional arguments for statistic.
    statistic_kwargs : dict, optional
        Additional keyword arguments for statistic.
    n_boostrap : int, default=5_000
        Number of bootstrap resamples.
    rng : int or np.random.Generator, optional
        Random number generator or seed. If None, uses default_rng(42).
    alternative : {'two-sided', 'greater', 'less', 'interval'}, default='two-sided'
        Alternative hypothesis:
        - 'two-sided': H1: statistic(test) != statistic(control)
        - 'greater':   H1: statistic(test) > statistic(control)
        - 'less':      H1: statistic(test) < statistic(control)
        - 'interval':  return bootstrap distribution for CI construction
    n_jobs : int, default=1
        Number of parallel jobs:
        - 1: no parallelism (safe for nested calls, default)
        - -1: use all available CPUs
        - n > 1: use n CPUs
        
        Performance recommendations based on benchmarks:
        - n_bootstrap <= 1000: always use n_jobs=1 (overhead too high)
        - n_bootstrap >= 5000 and 1000 <= sample_size <= 10000: use n_jobs=-1 (3-6x speedup)
        - Very large samples (>20k): overhead may dominate, test both
        
        Automatically detects nested parallelism (e.g., when called from
        DesignerIid.aa_sims) and disables inner parallelism to avoid
        process explosion.
    verbose : bool, default=False
        If True, show progress bar for parallel execution using tqdm.
        Only has effect when n_jobs > 1.
    
    Returns
    -------
    float or ndarray
        If alternative != 'interval': p-value
        If alternative == 'interval': bootstrap distribution of differences
    '''
    alt = _parse_alternative(alternative)

    test = np.asarray(test)
    control = np.asarray(control)

    assert test.ndim == 1
    assert control.ndim == 1

    # Initialize RNG
    if rng is None:
        rng = np.random.default_rng(42)
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)

    # Resolve n_jobs with nested parallelism detection
    n_jobs_actual = _resolve_n_jobs(n_jobs)

    # Sequential execution
    if n_jobs_actual == 1:
        # Generate bootstrap samples (resample with replacement)
        b_test = rng.choice(test, size=(n_bootstrap, len(test)), replace=True)
        b_control = rng.choice(control, size=(n_bootstrap, len(control)), replace=True)

        # Compute statistics for bootstrap samples
        stat_test = apply_statistic(
            samples=b_test,
            statistic=statistic,
            statistic_args=statistic_args,
            statistic_kwargs=statistic_kwargs,
            axis=1
        )
        stat_control = apply_statistic(
            samples=b_control,
            statistic=statistic,
            statistic_args=statistic_args,
            statistic_kwargs=statistic_kwargs,
            axis=1
        )

        boots = stat_test - stat_control
    else:
        # Parallel execution: split n_boostrap into chunks
        chunk_size = n_bootstrap // n_jobs_actual
        remainder = n_bootstrap % n_jobs_actual

        # Generate unique seeds for each worker
        seeds = rng.integers(0, 2**31, size=n_jobs_actual)
        chunk_sizes = [chunk_size + (1 if i < remainder else 0) for i in range(n_jobs_actual)]

        if verbose:
            ParallelClass = ProgressParallel(
                use_tqdm=True,
                total=n_jobs_actual,
                desc='Bootstrap chunks',
                unit='chunk',
                leave=True,
                n_jobs=n_jobs_actual,
                backend='loky',
                initializer=_worker_init_warnings
            )
        else:
            ParallelClass = Parallel(n_jobs=n_jobs_actual, backend='loky', initializer=_worker_init_warnings)

        # Compute chunks in parallel (thread limiting is inside _two_sample_test_chunk)
        chunks = ParallelClass(
            delayed(_two_sample_test_chunk)(
                test, control, None, chunk_sizes[i], seeds[i],
                statistic, statistic_args, statistic_kwargs, 'bootstrap'
            )
            for i in range(n_jobs_actual)
        )

        # Concatenate results from all workers
        boots = np.concatenate(chunks)

    # Return distribution for interval construction
    if alt == Alternative.INTERVAL:
        return boots
    
    # Compute p-value
    return _boot_pval(boots, alternative=alt)

def _one_sample_permutation_chunk(
    d_data: NDArray[np.floating],
    n_iterations: int,
    seed: int,
    statistic: Callable,
    statistic_args: Optional[Tuple],
    statistic_kwargs: Optional[Dict[str, Any]]
) -> NDArray[np.floating]:
    '''
    Worker function for parallel sign-permutation test.
    
    Generates a chunk of sign-flipped permutations and computes statistics.
    Thread limiting is applied to avoid NumPy BLAS oversubscription.
    
    Parameters
    ----------
    d_data : ndarray
        Original paired differences.
    n_iterations : int
        Number of permutations for this chunk.
    seed : int
        Random seed for this worker.
    statistic : callable
        Statistic function to apply.
    statistic_args : tuple or None
        Additional positional arguments for statistic.
    statistic_kwargs : dict or None
        Additional keyword arguments for statistic.
    
    Returns
    -------
    ndarray
        Permutation statistics for this chunk.
    '''
    with threadpool_limits(limits=1, user_api='blas'):
        rng = np.random.default_rng(seed)
        n = len(d_data)
        
        # Generate random signs {-1, +1}
        signs = rng.choice([-1, 1], size=(n_iterations, n))
        
        # Apply sign permutations (vectorized)
        d_perm = signs * d_data
        
        # Compute statistics for all permutations
        t_perm = apply_statistic(
            samples=d_perm,
            statistic=statistic,
            statistic_args=statistic_args,
            statistic_kwargs=statistic_kwargs,
            axis=1
        )
        
        return t_perm


def sign_permutation_test(
    d: ArrayLike,
    n_perm: int = 10_000,
    statistic: Callable = np.mean,
    statistic_args: Optional[Tuple] = None,
    statistic_kwargs: Optional[Dict[str, Any]] = None,
    alternative: Union[str, Alternative] = 'two-sided',
    rng: Optional[Union[int, np.random.Generator]] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> Union[float, NDArray[np.floating]]:
    '''
    Dependent (paired) sign-permutation test.

    Under H0, the signs of differences d_i are equally likely to be +1 or -1,
    i.e. the distribution of d is symmetric around zero. This test randomly flips
    the signs of differences and compares the observed statistic to the permutation
    distribution.

    Parameters
    ----------
    d : array-like
        Paired differences (e.g., test - control for matched pairs).
    n_perm : int, default=10_000
        Number of permutations to generate.
    statistic : callable, default=np.mean
        Statistic to compute on each permutation.
    statistic_args : tuple, optional
        Additional positional arguments for statistic.
    statistic_kwargs : dict, optional
        Additional keyword arguments for statistic.
    alternative : {'two-sided', 'greater', 'less', 'interval'}, default='two-sided'
        Alternative hypothesis:
        - 'two-sided': H1: E[d] != 0
        - 'greater':   H1: E[d] > 0
        - 'less':      H1: E[d] < 0
        - 'interval':  return permutation distribution for CI construction
    rng : int or np.random.Generator, optional
        Random number generator or seed. If None, uses default_rng(42).
    n_jobs : int, default=1
        Number of parallel jobs:
        - 1: no parallelism (safe for nested calls, default)
        - -1: use all available CPUs
        - n > 1: use n CPUs
        
        Performance recommendations based on benchmarks:
        - n_perm <= 1000: always use n_jobs=1 (overhead too high)
        - n_perm >= 5000 and sample_size >= 1000: use n_jobs=-1 (2-8x speedup)
        - Very large samples (>20k): overhead may dominate, test both
        
        Automatically detects nested parallelism (e.g., when called from
        DesignerIid.aa_sims) and disables inner parallelism to avoid
        process explosion.
    verbose : bool, default=False
        If True, show progress bar for parallel execution using tqdm.
        Only has effect when n_jobs > 1.

    Returns
    -------
    float or ndarray
        If alternative != 'interval': p-value
        If alternative == 'interval': array of permutation statistics for CI construction

    Notes
    -----
    This is a dependent (paired) test. For independent two-sample tests, use
    permutation_test_2s instead.
    
    Comparison with scipy.stats.permutation_test
    ---------------------------------------------
    scipy.stats.permutation_test does NOT have a direct equivalent for the sign test.
    - permutation_type='samples' performs resampling (bootstrap-like), not sign-flipping
    - permutation_type='pairings' is only for 2+ samples
    
    The classical sign test (randomly flipping signs of differences) is what this
    function implements. For scipy equivalent of paired data, you would use:
    
    scipy.stats.wilcoxon(test, control, alternative='two-sided')  # Wilcoxon signed-rank
    
    or for a parametric test:
    
    scipy.stats.ttest_rel(test, control, alternative='two-sided')  # Paired t-test

    Examples
    --------
    >>> d = np.array([1.2, -0.5, 2.1, 0.8, -0.3])  # paired differences
    >>> pval = sign_permutation_test(d, alternative='two-sided')
    >>> pval_greater = sign_permutation_test(d, alternative='greater')
    '''
    # Parse alternative
    alt = _parse_alternative(alternative)
    
    # Initialize RNG
    if rng is None:
        rng = np.random.default_rng(42)
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)
    
    d = np.asarray(d)
    n = d.shape[0]

    # Observed statistic
    t_obs = apply_statistic(
        samples=d,
        statistic=statistic,
        statistic_args=statistic_args,
        statistic_kwargs=statistic_kwargs
    )

    # Resolve n_jobs with nested parallelism detection
    n_jobs_actual = _resolve_n_jobs(n_jobs)
    
    # Sequential execution
    if n_jobs_actual == 1:
        # Generate random signs {-1, +1}
        signs = rng.choice([-1, 1], size=(n_perm, n))
        
        # Apply sign permutations (vectorized)
        d_perm = signs * d
        
        # Compute statistics for all permutations
        t_perm = apply_statistic(
            samples=d_perm,
            statistic=statistic,
            statistic_args=statistic_args,
            statistic_kwargs=statistic_kwargs,
            axis=1
        )
    else:
        # Parallel execution: split n_perm into chunks
        chunk_size = n_perm // n_jobs_actual
        remainder = n_perm % n_jobs_actual
        
        # Generate unique seeds for each worker
        seeds = rng.integers(0, 2**31, size=n_jobs_actual)
        chunk_sizes = [chunk_size + (1 if i < remainder else 0) for i in range(n_jobs_actual)]
        
        if verbose:
            ParallelClass = ProgressParallel(
                use_tqdm=True,
                total=n_jobs_actual,
                desc='Sign perm chunks',
                unit='chunk',
                leave=True,
                n_jobs=n_jobs_actual,
                backend='loky',
                initializer=_worker_init_warnings
            )
        else:
            ParallelClass = Parallel(n_jobs=n_jobs_actual, backend='loky', initializer=_worker_init_warnings)
        # Compute chunks in parallel (thread limiting is inside chunk function)

        chunks = ParallelClass(
            delayed(_one_sample_permutation_chunk)(
                d, chunk_sizes[i], seeds[i],
                statistic, statistic_args, statistic_kwargs
            )
            for i in range(n_jobs_actual)
        )
        
        # Concatenate results from all workers
        t_perm = np.concatenate(chunks)

    # Return null distribution for interval construction
    if alt == Alternative.INTERVAL:
        return t_perm
    
    # Compute p-value using unified function
    return _perm_pval(t_perm, t_obs, alt)

def permutation_test_2s(
    test: ArrayLike,
    control: ArrayLike,
    n_perm: int = 10_000,
    statistic: Callable = np.mean,
    statistic_args: Optional[Tuple] = None,
    statistic_kwargs: Optional[Dict[str, Any]] = None,
    alternative: Union[str, Alternative] = 'two-sided',
    rng: Optional[Union[int, np.random.Generator]] = None,
    n_jobs: int = 1,
    verbose: bool = False
) -> Union[float, NDArray[np.floating]]:
    '''
    Independent two-sample permutation test with optional parallelization.

    Under H0, all n_test + n_control observations are exchangeable, so randomly
    reassigning group labels should not change the test statistic. This test pools
    both samples, randomly permutes group assignments, and compares the observed
    difference to the permutation distribution.

    Parameters
    ----------
    test : array-like
        Test group observations.
    control : array-like
        Control group observations.
    n_perm : int, default=10_000
        Number of permutations to generate.
    statistic : callable, default=np.mean
        Statistic to compute on each group (difference is computed automatically).
    statistic_args : tuple, optional
        Additional positional arguments for statistic.
    statistic_kwargs : dict, optional
        Additional keyword arguments for statistic.
    alternative : {'two-sided', 'greater', 'less', 'interval'}, default='two-sided'
        Alternative hypothesis:
        - 'two-sided': H1: statistic(test) != statistic(control)
        - 'greater':   H1: statistic(test) > statistic(control)
        - 'less':      H1: statistic(test) < statistic(control)
        - 'interval':  return permutation distribution for CI construction
    rng : int or np.random.Generator, optional
        Random number generator or seed. If None, uses default_rng(42).
    n_jobs : int, default=1
        Number of parallel jobs for permutation computation.
        - n_jobs=1: sequential execution (no parallelism)
        - n_jobs=-1: use all available CPU cores
        - n_jobs>1: use specified number of cores
        
        Automatically detects nested parallelism (e.g., when called from within
        joblib.Parallel) and falls back to n_jobs=1 to avoid oversubscription.
    verbose : bool, default=False
        If True, show progress bar for parallel execution using tqdm.
        Only has effect when n_jobs > 1.

    Returns
    -------
    float or ndarray
        If alternative != 'interval': p-value
        If alternative == 'interval': array of permutation statistics for CI construction

    Notes
    -----
    Performance Recommendations
    ---------------------------
    Parallelism can provide speedups for large sample sizes and many permutations:
    
    - n_perm <= 1000: always use n_jobs=1 (overhead dominates)
    - n_perm >= 5000 and 5000 <= sample_size <= 20000: use n_jobs=-1 (2-4x speedup)
    - Very large samples (>50k): overhead may dominate again
    
    For small/medium samples (<5k), vectorization alone is usually sufficient.
    
    Comparison with scipy.stats.permutation_test
    ---------------------------------------------
    This function is equivalent to scipy.stats.permutation_test for independent samples:
    
    scipy.stats.permutation_test(
        (test, control),
        statistic=lambda x, y: np.mean(x) - np.mean(y),
        permutation_type='independent',  # or 'samples' (same for 2 groups)
        alternative='two-sided',
        n_resamples=10_000,
        random_state=42
    )

    Key differences:
    - scipy uses exact permutations for small n (typically < 8-10), Monte Carlo otherwise
    - this implementation is always Monte Carlo
      * Advantage: consistent behavior regardless of sample size
      * Advantage: more memory-efficient for medium/large n (when exact becomes infeasible)
      * Trade-off: for very small n, exact test is more precise (no sampling variability)
    - scipy supports vectorized statistic via axis= kwarg; this uses apply_statistic helper
    - scipy returns PermutationTestResult(statistic, pvalue, null_distribution)
    - this function returns only pvalue (or null distribution if alternative='interval')
    
    Both implementations give very similar p-values for typical use cases (n > 10).
    For very small n, exact test is more accurate but Monte Carlo is still acceptable.

    Examples
    --------
    >>> test = np.array([5.2, 6.1, 5.8, 6.3])
    >>> control = np.array([4.1, 4.5, 4.3, 4.8])
    >>> pval = permutation_test_2s(test, control, alternative='greater')
    
    >>> # Use parallelism for large samples
    >>> test_large = np.random.randn(10000)
    >>> control_large = np.random.randn(10000) + 0.1
    >>> pval = permutation_test_2s(test_large, control_large, n_perm=10000, n_jobs=-1)
    '''
    # Parse alternative
    alt = _parse_alternative(alternative)
    
    # Initialize RNG
    if rng is None:
        rng = np.random.default_rng(42)
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)
    
    test = np.asarray(test)
    control = np.asarray(control)
    
    n_test = len(test)
    n_control = len(control)
    n_total = n_test + n_control
    
    # Observed statistic (difference)
    t_obs_test = apply_statistic(
        samples=test,
        statistic=statistic,
        statistic_args=statistic_args,
        statistic_kwargs=statistic_kwargs
    )
    t_obs_control = apply_statistic(
        samples=control,
        statistic=statistic,
        statistic_args=statistic_args,
        statistic_kwargs=statistic_kwargs
    )
    t_obs = t_obs_test - t_obs_control
    
    # Pool all observations
    pooled = np.concatenate([test, control])
    
    # Resolve n_jobs (detect nested parallelism)
    n_jobs_actual = _resolve_n_jobs(n_jobs)
    
    # Parallel execution
    if n_jobs_actual > 1:
        # Split permutations into chunks for parallel processing
        chunk_size = n_perm // n_jobs_actual
        remainder = n_perm % n_jobs_actual
        
        # Generate unique seeds for each chunk
        base_seed = rng.integers(0, 2**31)
        
        # Execute chunks in parallel (thread limiting is inside _two_sample_test_chunk)
        tasks = []
        for i in range(n_jobs_actual):
            n_perm_chunk = chunk_size + (1 if i < remainder else 0)
            if n_perm_chunk > 0:
                tasks.append(
                    delayed(_two_sample_test_chunk)(
                        test, control, pooled, n_perm_chunk, base_seed + i,
                        statistic, statistic_args, statistic_kwargs, 'permutation'
                    )
                )
        if verbose:
            ParallelClass = ProgressParallel(
                use_tqdm=True,
                total=len(tasks),
                desc='Permutation chunks',
                unit='chunk',
                leave=True,
                n_jobs=n_jobs_actual,
                backend='loky',
                initializer=_worker_init_warnings
            )
        else:
            ParallelClass = Parallel(n_jobs=n_jobs_actual, backend='loky', initializer=_worker_init_warnings)

        results = ParallelClass(tasks)
        
        # Concatenate results from all chunks
        t_perm = np.concatenate(results)
    
    # Sequential execution (vectorized)
    else:
        # Generate permutations: randomly assign group labels (vectorized)
        # Create permutation matrix: each row is a random permutation of indices
        perm_matrix = np.argsort(rng.random((n_perm, n_total)), axis=1)
        
        # Split into test and control groups for all permutations
        # Shape: (n_perm, n_test) and (n_perm, n_control)
        perm_test = pooled[perm_matrix[:, :n_test]]
        perm_control = pooled[perm_matrix[:, n_test:]]
        
        # Compute statistics for all permuted test groups
        stat_test = apply_statistic(
            samples=perm_test,
            statistic=statistic,
            statistic_args=statistic_args,
            statistic_kwargs=statistic_kwargs,
            axis=1
        )
        
        # Compute statistics for all permuted control groups
        stat_control = apply_statistic(
            samples=perm_control,
            statistic=statistic,
            statistic_args=statistic_args,
            statistic_kwargs=statistic_kwargs,
            axis=1
        )
        
        # Compute differences (vectorized)
        t_perm = stat_test - stat_control
    
    # Return null distribution for interval construction
    if alt == Alternative.INTERVAL:
        return t_perm
    
    # Compute p-value using unified function
    return _perm_pval(t_perm, t_obs, alt)

def bucket_test_2s(
    test: ArrayLike,
    control: ArrayLike,
    pval_func: Callable[[NDArray[np.floating], NDArray[np.floating]], float],
    n_buckets: int = 100,
    statistic: Callable = np.mean,
    statistic_args: Optional[Tuple] = None,
    statistic_kwargs: Optional[Dict[str, Any]] = None,
    rng: Optional[Union[int, np.random.Generator]] = None
) -> float:
    '''
    Independent two-sample bucket (aggregation) test.

    Each sample is randomly split into ``n_buckets`` non-overlapping buckets.
    A ``statistic`` is computed within every bucket, producing ``n_buckets``
    aggregated values per group. The two resulting arrays of bucket statistics
    are then compared with a Welch (or Student) two-sample t-test.

    The bucket test is a variance-reduction / robustness technique: by
    aggregating raw observations into buckets, the per-bucket statistics become
    approximately normal (CLT), which makes a simple t-test on the bucketed
    values valid even for heavy-tailed or non-normal raw data. It is also much
    cheaper than bootstrap/permutation while giving comparable power.

    Parameters
    ----------
    test : array-like
        Test group observations.
    control : array-like
        Control group observations.
    pval_func : callable
        Function used to compute the p-value from the two arrays of bucket
        statistics. It is called as ``pval_func(test_buckets, control_buckets)``
        and must return a float p-value (e.g., a two-sample t-test or
        Mann-Whitney U test on the bucketed values). Required.
    n_buckets : int, default=100
        Number of buckets each group is split into. Must be >= 2 and not
        exceed the size of the smaller group.
    statistic : callable, default=np.mean
        Statistic to compute within each bucket (e.g., np.mean, np.median).
    statistic_args : tuple, optional
        Additional positional arguments for statistic.
    statistic_kwargs : dict, optional
        Additional keyword arguments for statistic.
    rng : int or np.random.Generator, optional
        Random number generator or seed used to shuffle observations into
        buckets. If None, uses default_rng(42).

    Returns
    -------
    float
        P-value computed by ``pval_func`` on the bucket statistics.

    Notes
    -----
    Observations that do not divide evenly among the buckets are distributed so
    that the first ``len(x) % n_buckets`` buckets receive one extra element.
    Bucket assignment is randomized to avoid ordering artefacts.

    Examples
    --------
    >>> test = np.random.exponential(scale=1.0, size=100_000)
    >>> control = np.random.exponential(scale=1.05, size=100_000)
    >>> pval = bucket_test_2s(
    ...     test, control,
    ...     pval_func=lambda t, c: stats.ttest_ind(t, c, equal_var=False).pvalue,
    ...     n_buckets=200,
    ... )

    >>> # Use a different p-value function on the bucket statistics
    >>> pval = bucket_test_2s(
    ...     test, control,
    ...     pval_func=lambda t, c: stats.mannwhitneyu(t, c).pvalue,
    ... )
    '''
    test = np.asarray(test)
    control = np.asarray(control)

    assert test.ndim == 1
    assert control.ndim == 1

    if n_buckets < 2:
        raise ValueError(f"n_buckets must be >= 2, got {n_buckets}")
    if n_buckets > min(len(test), len(control)):
        raise ValueError(
            f"n_buckets ({n_buckets}) cannot exceed the smaller group size "
            f"({min(len(test), len(control))})"
        )

    # Initialize RNG
    if rng is None:
        rng = np.random.default_rng(42)
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)

    def _bucket_statistics(x: NDArray[np.floating]) -> NDArray[np.floating]:
        # Randomly shuffle, then split into (almost) equal-sized buckets
        shuffled = x[rng.permutation(len(x))]
        buckets = np.array_split(shuffled, n_buckets)
        return np.array([
            apply_statistic(
                samples=bucket,
                statistic=statistic,
                statistic_args=statistic_args,
                statistic_kwargs=statistic_kwargs
            )
            for bucket in buckets
        ])

    test_buckets = _bucket_statistics(test)
    control_buckets = _bucket_statistics(control)

    result = pval_func(test_buckets, control_buckets)

    return float(result)

# ------------------------------------------------------------------------------------------------ #
#                                      Multiple Comparisons                                        #
# ------------------------------------------------------------------------------------------------ #

def no_multiple_correction(
    pvalues: NDArray[np.floating],
    copy: bool = False
) -> NDArray[np.floating]:
    '''
    No correction for multiple comparisons.
    
    Returns p-values without any adjustment. Useful as a baseline or when
    correction is not needed.
    
    Parameters
    ----------
    pvalues : ndarray
        Array of p-values.
    copy : bool, default=False
        If True, return a copy of the input array.
    
    Returns
    -------
    ndarray
        Unadjusted p-values.
    '''
    if copy:
        pvalues = pvalues.copy()
    return pvalues

def bonferroni(
    pvalues: NDArray[np.floating]
) -> NDArray[np.floating]:
    '''
    Bonferroni correction for multiple comparisons.
    
    Controls the family-wise error rate (FWER) by adjusting p-values:
    p_adj = min(p * m, 1), where m is the number of tests.
    
    Parameters
    ----------
    pvalues : ndarray
        Array of p-values.
    
    Returns
    -------
    ndarray
        Bonferroni-adjusted p-values.
    
    Notes
    -----
    Most conservative method. Controls FWER under any dependency structure.
    '''
    m = len(pvalues)
    return np.minimum(pvalues * m, 1.0)

def holm_bonferroni(
    pvalues: NDArray[np.floating]
) -> NDArray[np.floating]:
    '''
    Holm-Bonferroni step-down procedure.
    
    Controls the family-wise error rate (FWER) with more power than Bonferroni.
    Sequentially tests hypotheses from smallest to largest p-value.
    
    Parameters
    ----------
    pvalues : ndarray
        Array of p-values.
    
    Returns
    -------
    ndarray
        Holm-Bonferroni adjusted p-values.
    
    Notes
    -----
    Controls FWER under any dependency structure. More powerful than Bonferroni.
    '''
    m = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    adjusted = np.empty(m)
    cummax = 0.0
    for i in range(m):
        val = sorted_p[i] * (m - i)
        cummax = max(cummax, val)
        adjusted[order[i]] = min(cummax, 1.0)
    return adjusted

def hochberg(
    pvalues: NDArray[np.floating]
) -> NDArray[np.floating]:
    '''
    Hochberg step-up procedure.
    
    Controls the family-wise error rate (FWER) under independence or positive
    dependence. More powerful than Holm-Bonferroni under these conditions.
    
    Parameters
    ----------
    pvalues : ndarray
        Array of p-values.
    
    Returns
    -------
    ndarray
        Hochberg-adjusted p-values.
    
    Notes
    -----
    Controls FWER under independence or positive dependence.
    More powerful than Holm-Bonferroni but requires independence assumption.
    '''
    m = len(pvalues)
    order = np.argsort(pvalues)[::-1]  # descending order
    sorted_p = pvalues[order]
    adjusted = np.empty(m)
    cummin = 1.0
    for i in range(m):
        rank_asc = m - i  # rank in ascending order
        val = sorted_p[i] * (m - rank_asc + 1)
        cummin = min(cummin, val)
        adjusted[order[i]] = min(cummin, 1.0)
    return adjusted

def benjamini_hochberg(
    pvalues: NDArray[np.floating]
) -> NDArray[np.floating]:
    '''
    Benjamini-Hochberg (BH) procedure.
    
    Controls the false discovery rate (FDR) under independence or positive
    dependence. Less conservative than FWER-controlling methods.
    
    Parameters
    ----------
    pvalues : ndarray
        Array of p-values.
    
    Returns
    -------
    ndarray
        Benjamini-Hochberg adjusted p-values.
    
    Notes
    -----
    Controls FDR under independence or positive dependence (PRDS).
    Most commonly used FDR-controlling procedure.
    '''
    m = len(pvalues)
    order = np.argsort(pvalues)[::-1]  # descending order
    sorted_p = pvalues[order]
    adjusted = np.empty(m)
    cummin = 1.0
    for i in range(m):
        rank = m - i  # rank in ascending order
        val = sorted_p[i] * m / rank
        cummin = min(cummin, val)
        adjusted[order[i]] = min(cummin, 1.0)
    return adjusted

def benjamini_yekutieli(
    pvalues: NDArray[np.floating],
    alpha: float = 0.05
) -> NDArray[np.floating]:
    '''
    Benjamini-Yekutieli (BY) procedure.
    
    Controls the false discovery rate (FDR) under arbitrary dependence.
    More conservative than Benjamini-Hochberg but works for any dependency.
    
    Parameters
    ----------
    pvalues : ndarray
        Array of p-values.
    alpha : float, default=0.05
        Significance level (not used, kept for API consistency).
    
    Returns
    -------
    ndarray
        Benjamini-Yekutieli adjusted p-values.
    
    Notes
    -----
    Controls FDR under arbitrary dependence structure.
    Uses harmonic number c(m) = sum(1/i for i=1..m) as correction factor.
    '''
    m = len(pvalues)
    c_m = np.sum(1.0 / np.arange(1, m + 1))  # harmonic number
    order = np.argsort(pvalues)[::-1]
    sorted_p = pvalues[order]
    adjusted = np.empty(m)
    cummin = 1.0
    for i in range(m):
        rank = m - i
        val = sorted_p[i] * m * c_m / rank
        cummin = min(cummin, val)
        adjusted[order[i]] = min(cummin, 1.0)
    return adjusted

def benjamini_krieger_yekutieli(
    pvalues: NDArray[np.floating],
    alpha: float = 0.05
) -> NDArray[np.floating]:
    '''
    Benjamini-Krieger-Yekutieli (BKY) two-stage adaptive procedure.
    
    Adaptive FDR-controlling procedure that estimates the number of true nulls
    and adjusts the BH procedure accordingly. More powerful than BH when many
    hypotheses are true nulls.
    
    Parameters
    ----------
    pvalues : ndarray
        Array of p-values.
    alpha : float, default=0.05
        Target FDR level (used in the adaptive procedure).
    
    Returns
    -------
    ndarray
        BKY-adjusted p-values (new array, input unchanged).
    
    Notes
    -----
    Two-stage procedure:
    1. Apply BH at level alpha/(1+alpha) to estimate number of rejections r
    2. If r > 0, estimate m0 = m - r and apply BH at level alpha*m/m0
    3. If r = 0, return BH adjusted p-values
    
    Controls FDR under independence. More powerful than BH when proportion
    of true nulls is high.
    
    References
    ----------
    Benjamini, Y., Krieger, A. M., & Yekutieli, D. (2006).
    Adaptive linear step-up procedures that control the false discovery rate.
    Biometrika, 93(3), 491-507.
    '''
    pvalues = np.asarray(pvalues)
    m = len(pvalues)
    
    # Stage 1: Apply BH at level alpha/(1+alpha)
    alpha_1 = alpha / (1.0 + alpha)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    
    # Find number of rejections in stage 1.
    # BH is a step-UP procedure: we need the LARGEST rank k such that
    # p(k) <= k * alpha_1 / m.  Stopping at the first failure (step-DOWN)
    # would undercount rejections when a gap exists in the p-value sequence.
    r = 0
    for i in range(m):
        if sorted_p[i] <= (i + 1) * alpha_1 / m:
            r = i + 1
    
    # Stage 2: Adaptive adjustment
    if r == 0:
        # No rejections in stage 1, return standard BH
        return benjamini_hochberg(pvalues)
    
    # Estimate number of true nulls: m0 = m - r
    m0_hat = m - r
    
    # Apply BH with adjusted level: alpha * m / m0_hat
    adjusted = np.empty(m)
    cummin = 1.0
    order_desc = np.argsort(pvalues)[::-1]
    sorted_p_desc = pvalues[order_desc]
    
    for i in range(m):
        rank = m - i  # rank in ascending order
        val = sorted_p_desc[i] * m0_hat / rank
        cummin = min(cummin, val)
        adjusted[order_desc[i]] = min(cummin, 1.0)
    
    return adjusted

def sidak(
    pvalues: NDArray[np.floating]
) -> NDArray[np.floating]:
    '''
    Šidák correction for multiple comparisons.
    
    Controls the family-wise error rate (FWER) under independence.
    Formula: p_adj = 1 - (1 - p)^m. More accurate than Bonferroni under independence.
    
    Parameters
    ----------
    pvalues : ndarray
        Array of p-values.
    
    Returns
    -------
    ndarray
        Šidák-adjusted p-values.
    
    Notes
    -----
    Controls FWER under independence. More powerful than Bonferroni when
    tests are independent, but requires independence assumption.
    '''
    m = len(pvalues)
    return 1.0 - (1.0 - pvalues) ** m

def holm_sidak(
    pvalues: NDArray[np.floating]
) -> NDArray[np.floating]:
    '''
    Holm-Šidák step-down procedure.
    
    Controls the family-wise error rate (FWER) under independence.
    Combines Holm's step-down approach with Šidák correction.
    
    Parameters
    ----------
    pvalues : ndarray
        Array of p-values.
    
    Returns
    -------
    ndarray
        Holm-Šidák adjusted p-values.
    
    Notes
    -----
    Controls FWER under independence. More powerful than Holm-Bonferroni
    when tests are independent.
    '''
    m = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    adjusted = np.empty(m)
    cummax = 0.0
    for i in range(m):
        val = 1.0 - (1.0 - sorted_p[i]) ** (m - i)
        cummax = max(cummax, val)
        adjusted[order[i]] = min(cummax, 1.0)
    return adjusted


CORRECTION_METHODS = {
    'no_correction': no_multiple_correction,
    'bonferroni': bonferroni,
    'holm_bonferroni': holm_bonferroni,
    'hochberg': hochberg,
    'benjamini_hochberg': benjamini_hochberg,
    'benjamini_yekutieli': benjamini_yekutieli,
    'benjamini_krieger_yekutieli': benjamini_krieger_yekutieli,
    'sidak': sidak,
    'holm_sidak': holm_sidak,
}
"""
Dictionary of available multiple testing correction methods.

Keys
----
- 'no_correction': No adjustment
- 'bonferroni': Bonferroni correction (FWER)
- 'holm_bonferroni': Holm-Bonferroni step-down (FWER)
- 'hochberg': Hochberg step-up (FWER, requires independence)
- 'benjamini_hochberg': Benjamini-Hochberg (FDR)
- 'benjamini_yekutieli': Benjamini-Yekutieli (FDR, arbitrary dependence)
- 'benjamini_krieger_yekutieli': Benjamini-Krieger-Yekutieli adaptive (FDR, more powerful)
- 'sidak': Šidák correction (FWER, requires independence)
- 'holm_sidak': Holm-Šidák step-down (FWER, requires independence)

Notes
-----
FWER = Family-Wise Error Rate (probability of at least one false positive)
FDR = False Discovery Rate (expected proportion of false positives among rejections)
"""

def multiple_pvalue_correction(
    pvalues: NDArray[np.floating],
    method: Union[str, Callable] = "bonferroni",
    **kwargs
) -> NDArray[np.floating]:
    '''
    Apply multiple testing correction to p-values.
    
    Adjusts p-values to control for multiple comparisons using various methods
    that control either the family-wise error rate (FWER) or false discovery
    rate (FDR).
    
    Parameters
    ----------
    pvalues : ndarray
        Array of p-values to adjust.
    method : str, default='bonferroni'
        Correction method to use. Available methods:
        - 'no_correction': No adjustment
        - 'bonferroni': Bonferroni correction (FWER)
        - 'holm_bonferroni': Holm-Bonferroni step-down (FWER)
        - 'hochberg': Hochberg step-up (FWER, requires independence)
        - 'benjamini_hochberg': Benjamini-Hochberg (FDR)
        - 'benjamini_yekutieli': Benjamini-Yekutieli (FDR, arbitrary dependence)
        - 'benjamini_krieger_yekutieli': Benjamini-Krieger-Yekutieli adaptive (FDR, more powerful)
        - 'sidak': Šidák correction (FWER, requires independence)
        - 'holm_sidak': Holm-Šidák step-down (FWER, requires independence)
    
    Returns
    -------
    ndarray
        Adjusted p-values.
    
    Raises
    ------
    KeyError
        If method is not recognized.
    
    Examples
    --------
    >>> pvals = np.array([0.01, 0.04, 0.03, 0.005])
    >>> multiple_pvalue_correction(pvals, method='bonferroni')
    array([0.04, 0.16, 0.12, 0.02])
    
    >>> multiple_pvalue_correction(pvals, method='benjamini_hochberg')
    array([0.02, 0.04, 0.04, 0.02])
    
    Notes
    -----
    Choice of method depends on:
    - FWER vs FDR control: FWER is more conservative
    - Independence assumption: some methods require independent tests
    - Power: FDR methods generally more powerful than FWER methods
    
    Recommended methods:
    - Bonferroni or Holm-Bonferroni: when FWER control is critical
    - Benjamini-Hochberg: when FDR control is acceptable (most common)
    - Benjamini-Yekutieli: when tests are dependent and FDR control needed
    '''
    method = method or 'no_correction'
    if method in CORRECTION_METHODS:
        return CORRECTION_METHODS[method](pvalues, **kwargs)
    return method(pvalues, **kwargs)

def delta_method_pvalue(
    test: Tuple[NDArray, NDArray],
    control: Tuple[NDArray, NDArray]
) -> Tuple[float, float]:
    """
    Two-sample test for a ratio metric using the delta method.

    The variance of the per-group ratio ``mean(num) / mean(denom)`` is
    approximated via the delta method, and the difference between the test and
    control ratios is compared with a two-sided normal (z) test.

    Parameters
    ----------
    test : tuple of ndarray
        ``(num_array, denom_array)`` for the test group.
    control : tuple of ndarray
        ``(num_array, denom_array)`` for the control group.

    Returns
    -------
    pvalue : float
        Two-sided p-value of the test.
    delta : float
        Difference between the test and control ratio metrics.
    """
    def group_stats(x, y):
        n = len(x)
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        var_x  = np.var(x, ddof=1)
        var_y  = np.var(y, ddof=1)
        cov_xy = np.cov(x, y)[0, 1]
        metric = mean_x / mean_y
        var_metric = (
            var_x / mean_y**2
            + mean_x**2 / mean_y**4 * var_y
            - 2 * mean_x / mean_y**3 * cov_xy
        ) / n
        return metric, var_metric

    metric_t, var_t = group_stats(*test)
    metric_c, var_c = group_stats(*control)

    delta     = metric_t - metric_c
    statistic = delta / np.sqrt(var_t + var_c)
    pvalue    = (1 - stats.norm.cdf(np.abs(statistic))) * 2
    return pvalue, delta
