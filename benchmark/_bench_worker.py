"""
_bench_worker.py
================
Helper subprocess for cold-pool n_jobs=-1 benchmarks.
Called by mcab_benchmark.py; prints elapsed seconds to stdout.

Usage:
    python _bench_worker.py <scenario> <n_sims> <data_size> <effect> <seed>
    scenario: aa | ab | multi
"""

import sys
import time
import warnings
import matplotlib; matplotlib.use('Agg')
import numpy as np
from scipy import stats

warnings.filterwarnings('ignore')

def main():
    scenario  = sys.argv[1]          # aa | ab | multi
    n_sims    = int(sys.argv[2])
    data_size = int(sys.argv[3])
    effect    = float(sys.argv[4])
    seed      = int(sys.argv[5])

    from mcab import RandomData, DesignerIid, BenchMarker

    rd   = RandomData(seed=seed)
    pval = lambda t, c: stats.ttest_ind(t, c, equal_var=False).pvalue

    if scenario in ('aa', 'ab'):
        data = rd.exponential_data(size=data_size, scale=1_000_000)
        des  = DesignerIid(target=data, alpha=0.05,
                           effect_sizer='percent', pval_func=pval)
        t0 = time.perf_counter()
        if scenario == 'aa':
            des.aa_sims(n_sims=n_sims, verbose=False, plot=False, n_jobs=-1)
        else:
            des.calculate_some_power(effect=effect, n_sims=n_sims,
                                     verbose=False, n_jobs=-1)
        print(f"{time.perf_counter() - t0:.4f}", flush=True)

    elif scenario == 'multi':
        scales = [1_000_000, 500_000, 2_000_000, 3_000_000]
        data_list = [rd.exponential_data(size=data_size, scale=s) for s in scales]
        designers = [DesignerIid(target=d, alpha=0.05,
                                 effect_sizer='percent', pval_func=pval)
                     for d in data_list]
        bm = BenchMarker(designers)
        t0 = time.perf_counter()
        bm.multi_test_1_type_error(multi_corrections='holm_bonferroni',
                                   n_sims=n_sims, verbose=False, n_jobs=-1)
        print(f"{time.perf_counter() - t0:.4f}", flush=True)

if __name__ == '__main__':
    main()
