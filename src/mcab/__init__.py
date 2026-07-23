"""
mcab - Monte Carlo A/B Testing Library
========================================

A comprehensive Python library for A/B testing design and analysis using Monte Carlo simulations.

Main Components
---------------
- RandomData: Generate synthetic data for testing
- AaDataIid: Handle independent (user-aggregated) data
- AaDataRatio: Handle ratio metrics (CTR, average check, etc.)
- AaDataDept1s: Handle dependent one-sample data with an artificial control
- DesignerIid: Design A/B tests for independent data
- DesignerRatio: Design A/B tests for ratio metrics
- DesignerRatioLin: Design A/B tests for linearized ratio metrics
- DesignerDept1s: Design A/B tests for dependent one-sample data
- BenchMarker: Compare multiple A/B test designs

Statistical Tests
-----------------
- bootstrap_1s, bootstrap_2s: Bootstrap hypothesis tests
- sign_permutation_test: Paired permutation test
- permutation_test_2s: Two-sample permutation test
- multiple_pvalue_correction: Multiple testing corrections

Example Usage
-------------
```python
>>> from mcab import RandomData, AaDataIid, DesignerIid
>>> from scipy import stats
>>> 
>>> # Generate data
>>> rd = RandomData(seed=42)
>>> data = rd.normal_data(size=10000, mean=100, std=30)
>>> 
>>> # Create designer
>>> target = AaDataIid(data)
>>> designer = DesignerIid(target, alpha=0.05)
>>> 
>>> # Run AA simulations
>>> pvals = designer.aa_sims(
...     pval_func=lambda t, c: stats.ttest_ind(t, c).pvalue,
...     n_sims=1000
... )
>>> 
>>> # Find MDE
>>> mde_result = designer.find_mde(
...     pval_func=lambda t, c: stats.ttest_ind(t, c).pvalue,
...     target_power=0.8
... )
```
"""

__version__ = "0.1.0"
__author__ = "Sergey Lastochkin"

# Core classes
from .core import (
    RandomData,
    AaDataIid,
    AaDataRatio,
    AaDataDept1s,
    DesignerIid,
    DesignerRatio,
    DesignerRatioLin,
    DesignerDept1s,
    BenchMarker,
)

from .effects import (
    PercentEffectSizer,
    ConstantEffectSizer,
    CustomEffectSizer,
    FunctionEffectSizer
)

# Utility classes
from .utils import ProgressParallel

# Statistical functions
from .mstats import (
    # Enums
    Alternative,
    # Utilities
    apply_statistic,
    # Bootstrap tests
    bootstrap_1s,
    bootstrap_2s,
    # Permutation tests
    sign_permutation_test,
    permutation_test_2s,
    # Bucket test
    bucket_test_2s,
    # Multiple testing corrections
    multiple_pvalue_correction,
    CORRECTION_METHODS,
    bonferroni,
    holm_bonferroni,
    hochberg,
    benjamini_hochberg,
    benjamini_yekutieli,
    benjamini_krieger_yekutieli,
    sidak,
    holm_sidak,
    no_multiple_correction,
    delta_method_pvalue
)

# Transformers
from .transform import (
    Linearizer,
    CUPAC,
    CUPED,
    PostStratification,
    coef_pvalue_table,
)

__all__ = [
    # Version
    '__version__',
    # Core classes
    'RandomData',
    'AaDataIid',
    'AaDataRatio',
    'AaDataDept1s',
    'DesignerIid',
    'DesignerRatio',
    'DesignerRatioLin',
    'DesignerDept1s',
    'BenchMarker',
    # Effects
    'PercentEffectSizer',
    'ConstantEffectSizer',
    'CustomEffectSizer',
    'FunctionEffectSizer',
    # Utilities
    'ProgressParallel',
    # Enums
    'Alternative',
    # Statistical utilities
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
    # Multiple testing
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
    'no_multiple_correction',
    # Transformers
    'Linearizer',
    'CUPAC',
    'CUPED',
    'PostStratification',
    'coef_pvalue_table'
]
