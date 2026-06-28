from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tqdm.auto import tqdm

from typing import List, Tuple, Dict, Optional, Union, Callable, Literal
from numpy.typing import NDArray

from .core import (
    RandomData,
    AaDataIid,
    AaDataRatio,
    DesignerIid,
    DesignerRatio,
    DesignerRatioLin
)

from .transform import (
    CUPED,
    CUPAC,
    PostStratification
)

def _make_transform(transformer, X, y):
    if isinstance(y, tuple):
        num, den = y
        num_mod = num - transformer().fit_predict(X, num)
        den_mod = den - transformer().fit_predict(X, den)
        return (num_mod, den_mod)
    return y - transformer().fit_predict(X, y)

def make_cuped(X, y):
    return _make_transform(CUPED, X, y)

def make_cupac(X, y):
    return _make_transform(CUPAC, X, y)

def make_post_stratification(X, y):
    return _make_transform(PostStratification, X, y)

def IidExample(
        size,
        proportion: bool = False,
        effect_sizer: Literal['percent', 'const', 'custom'] = 'percent',
        transformer_name: Optional[Literal['cuped', 'cupac', 'strat']] = None,
        pval_func: Optional[Callable] = None,
        pval_func_name: Optional[str] = None,
        seed: int = 42
    ):
        assert transformer_name in ['cuped', 'cupac', 'strat', None], 'transformer must be one of cuped, cupac, strat, or None'
        
        des_name = 'iid'
        
        data_generator = RandomData(seed)

        is_prop = None

        if proportion:
            data_raw = data_generator.proportion_data(size=size, p=0.70)
            is_prop = 'prop'
        else:
            data_raw = data_generator.exponential_data(size=size)

        transformer = None
        covariates  = None

        if transformer_name in ('cuped', 'cupac'):
            cuped_cov = (data_raw * 0.8) + np.random.normal(0, data_raw.std(), size=size)

        if transformer_name in ('cupac', 'strat'):
            strat_cov = (data_raw > data_raw.mean()).astype(int)

        if transformer_name == 'cuped':
            transformer = make_cuped
            covariates = cuped_cov
        elif transformer_name == 'strat':
            transformer = make_post_stratification
            covariates = strat_cov
        elif transformer_name == 'cupac':
            transformer = make_cupac
            covariates = np.column_stack([
                cuped_cov,
                strat_cov
            ])

        if (pval_func is not None) and (pval_func_name is None):
            pval_func_name = 'custom'
        elif pval_func is None:
            pval_func = lambda test, control: stats.ttest_ind(test, control).pvalue
            pval_func_name = 'ttest_ind'

        name_fields = [
            des_name,
            is_prop,
            effect_sizer,
            transformer_name,
            pval_func_name,
        ]
        des_name_full = '_'.join([x for x in name_fields if x])

        data = AaDataIid(
            data_raw,
            transformer=transformer,
            covariates=covariates,
            proportion=proportion
        )
        
        return DesignerIid(
            target=data,
            name=des_name_full,
            pval_func=pval_func,
            effect_sizer=effect_sizer
        )

def RatioExample(
        size,
        proportion: bool = False,
        linearize: bool = False,
        effect_sizer: Literal['percent', 'const', 'custom'] = 'percent',
        transformer_name: Optional[Literal['cuped', 'cupac', 'strat']] = None,
        pval_func: Optional[Callable] = None,
        pval_func_name: Optional[str] = None,
        seed: int = 42
    ):
        assert transformer_name in ['cuped', 'cupac', 'strat', None], 'transformer must be one of cuped, cupac, strat, or None'

        des_name = 'ratio'

        data_generator = RandomData(seed)

        is_prop = None
        lin_name = None

        if proportion:
            numerator, denominator = data_generator.ratio_data_ctr(size=size)
            is_prop = 'prop'
        else:
            numerator, denominator = data_generator.ratio_data_avg_bill(size=size)

        data_raw = (numerator, denominator)

        transformer = None
        covariates  = None

        # AaDataRatio requires covariates as a 2-D matrix, so single covariates
        # are reshaped to (size, 1) and multiple ones are column-stacked.
        if transformer_name in ('cuped', 'cupac'):
            cuped_cov = (numerator * 0.8) + np.random.normal(0, numerator.std(), size=size)

        if transformer_name in ('cupac', 'strat'):
            strat_cov = (numerator > numerator.mean()).astype(int)

        if transformer_name == 'cuped':
            transformer = make_cuped
            covariates = cuped_cov.reshape(-1, 1)
        elif transformer_name == 'strat':
            transformer = make_post_stratification
            covariates = strat_cov.reshape(-1, 1)
        elif transformer_name == 'cupac':
            transformer = make_cupac
            covariates = np.column_stack([
                cuped_cov,
                strat_cov
            ])

        # Linearized ratio metrics use the dedicated DesignerRatioLin, whose
        # pval_func receives 1-D linearized arrays (like the iid case); the
        # plain DesignerRatio instead receives (numerator, denominator) tuples.
        if linearize:
            designer_class = DesignerRatioLin
            lin_name = 'lin'
        else:
            designer_class = DesignerRatio

        if (pval_func is not None) and (pval_func_name is None):
            pval_func_name = 'custom'
        elif pval_func is None:
            if linearize:
                pval_func = lambda test, control: stats.ttest_ind(test, control).pvalue
            else:
                pval_func = lambda test, control: stats.ttest_ind(
                    test[0] / test[1], control[0] / control[1]
                ).pvalue
            pval_func_name = 'ttest_ind'

        name_fields = [
            des_name,
            is_prop,
            lin_name,
            effect_sizer,
            transformer_name,
            pval_func_name,
        ]
        des_name_full = '_'.join([x for x in name_fields if x])

        data = AaDataRatio(
            data_raw,
            transformer=transformer,
            covariates=covariates,
            proportion=proportion
        )

        return designer_class(
            target=data,
            name=des_name_full,
            pval_func=pval_func,
            effect_sizer=effect_sizer
        )

def _get_exact_tests_results():
    class TData:
        def __init__(self, name, value):
            self.name  = name
            self.value = value

        def __repr__(self):
            return f'TData({self.name})'

    data_size = 20

    strat_1 = TData('strat_1', pd.Series(['one' for x in range(data_size // 2)] + ['two' for x in range(data_size // 2)], name='strat').to_frame())
    strat_2 = TData('strat_2', pd.Series([1. for x in range(data_size // 2)] + [0. for x in range(data_size // 2)], name='strat').to_frame())

    iid_data = TData('iid_data', AaDataIid(pd.Series(np.arange(1, data_size+1), name='data')))
    iid_data_prop = TData(
        'iid_data_prop',
        [
            y
            for x in [[1, 0] for x in range(data_size//2)]
            for y in x
        ]
    )
    iid_data_prop.value[data_size//2] = 0
    iid_data_prop.value = pd.Series(iid_data_prop.value, name='data')
    iid_data_prop.value = AaDataIid(iid_data_prop.value, proportion=True)

    ratio_data_big   = iid_data.value.data_raw+(np.maximum(np.arange(1, (data_size*10)+1, step=10) - 1, 1))
    ratio_data_small = iid_data.value.data_raw
    ratio_data       = TData('ratio_data',      AaDataRatio((ratio_data_big,   ratio_data_small)))
    ratio_data_prop  = TData('ratio_data_prop', AaDataRatio((ratio_data_small, ratio_data_big), proportion=True))

    iid_cov        = TData('iid_cov',        pd.Series(iid_data.value.data_raw * 0.8, name='iid_cov'))
    ratio_cov      = TData('ratio_cov',      pd.Series(ratio_data.value.data_raw[0] * 0.8, name='ratio_cov'))
    ratio_prop_cov = TData('ratio_prop_cov', pd.Series(ratio_data_prop.value.data_raw[0] * 0.8, name='ratio_prop_cov'))

    iid_X1        = TData('iid_X1',        pd.concat([iid_cov.value,        strat_1.value], axis=1))
    iid_X2        = TData('iid_X2',        pd.concat([iid_cov.value,        strat_2.value], axis=1))
    ratio_X1      = TData('ratio_X1',      pd.concat([ratio_cov.value,      strat_1.value], axis=1))
    ratio_X2      = TData('ratio_X2',      pd.concat([ratio_cov.value,      strat_2.value], axis=1))
    ratio_prop_X1 = TData('ratio_prop_X1', pd.concat([ratio_prop_cov.value, strat_1.value], axis=1))
    ratio_prop_X2 = TData('ratio_prop_X2', pd.concat([ratio_prop_cov.value, strat_2.value], axis=1))

    pval_func_iid   = TData('pval_func_iid',   lambda test, control: stats.ttest_ind(test, control, equal_var=False).pvalue)
    pval_func_ratio = TData('pval_func_ratio', lambda test, control: stats.ttest_ind(test[0]/test[1], control[0]/control[1], equal_var=False).pvalue)
    pval_func_dept  = TData('pval_func_dept',  lambda test: stats.ttest_1samp(test, 0).pvalue)

    designer_iid            = TData('designer_iid',            DesignerIid)
    designer_iid_prop       = TData('designer_iid_prop',       DesignerIid)
    designer_ratio          = TData('designer_ratio',          DesignerRatio)
    designer_ratio_lin      = TData('designer_ratio_lin',      DesignerRatioLin)
    designer_ratio_prop     = TData('designer_ratio_prop',     DesignerRatio)
    deisgner_ratio_prop_lin = TData('designer_ratio_prop_lin', DesignerRatioLin)

    DESIGNERS = {
        designer_iid:            {'data':iid_data,        'pval_func':pval_func_iid,   'X':{'none':[None], 'cuped':[iid_cov],        'cupac':[iid_X1, iid_X2],               'post_stratification':[strat_1, strat_2]}},
        designer_iid_prop:       {'data':iid_data_prop,   'pval_func':pval_func_iid,   'X':{'none':[None], 'cuped':[iid_cov],        'cupac':[iid_X1, iid_X2],               'post_stratification':[strat_1, strat_2]}},
        designer_ratio:          {'data':ratio_data,      'pval_func':pval_func_ratio, 'X':{'none':[None], 'cuped':[ratio_cov],      'cupac':[ratio_X1, ratio_X2],           'post_stratification':[strat_1, strat_2]}},
        designer_ratio_lin:      {'data':ratio_data,      'pval_func':pval_func_ratio, 'X':{'none':[None], 'cuped':[ratio_cov],      'cupac':[ratio_X1, ratio_X2],           'post_stratification':[strat_1, strat_2]}},
        designer_ratio_prop:     {'data':ratio_data_prop, 'pval_func':pval_func_ratio, 'X':{'none':[None], 'cuped':[ratio_prop_cov], 'cupac':[ratio_prop_X1, ratio_prop_X2], 'post_stratification':[strat_1, strat_2]}},
        deisgner_ratio_prop_lin: {'data':ratio_data_prop, 'pval_func':pval_func_ratio, 'X':{'none':[None], 'cuped':[ratio_prop_cov], 'cupac':[ratio_prop_X1, ratio_prop_X2], 'post_stratification':[strat_1, strat_2]}}
    }

    EFFECT_SIZERS = [
        TData('AA',                  ('percent',  0.00, None)),
        TData('PercentEffectSizer',  ('percent',  0.01, None)),
        TData('ConstantEffectSizer', ('const',    0.01, None)),
        TData('CustomEffectSizer',   ('custom',   1.00, lambda x: x*1.01)),
    ]

    TRANSFORMERS = {
        'none':                None,
        'cuped':               make_cuped,
        'cupac':               make_cupac,
        'post_stratification': make_post_stratification,
    }

    tests_res = []
    tqdm_total = (
        sum([(len(z)*len(EFFECT_SIZERS)) for x in DESIGNERS.values() for z in x['X'].values()]) # all covariates of all transformers with all effects
    )
    with tqdm(total=tqdm_total, desc='Tests', unit='test') as pbar:
        for des_num, (des, designer_data) in enumerate(DESIGNERS.items()):

            des_name  = des.name
            des_class = des.value

            des_data_name = designer_data['data'].name
            des_data      = designer_data['data'].value

            des_pval_func_name = designer_data['pval_func'].name
            des_pval_func      = designer_data['pval_func'].value

            des_X = designer_data['X']

            for tr_i, (trans_name, trans_cov_list) in enumerate(des_X.items()):

                trans_func = TRANSFORMERS[trans_name]

                for trans_cov_i, trans_cov in enumerate(trans_cov_list):

                    trans_cov_name = None
                    if trans_cov is not None:
                        trans_cov_name = trans_cov.name

                    for efs_i, efs_data in enumerate(EFFECT_SIZERS):
                        efs_name = efs_data.name
                        effect_sizer, effect, custom_data_func = efs_data.value
                        if custom_data_func is None:
                            custom_data = None
                        else:
                            if isinstance(des_data.data_raw, tuple):
                                custom_data = (custom_data_func(des_data.data_raw[0]), des_data.data_raw[1])
                            else:
                                custom_data = custom_data_func(des_data.data_raw)

                        if (trans_func is not None) or (trans_cov is not None):
                            des_data = deepcopy(des_data)
                        
                        if trans_func is not None:
                            des_data.transformer = trans_func

                        if trans_cov is not None:
                            des_data.covariates = trans_cov.value

                        designer = des_class(
                            des_data,
                            pval_func=des_pval_func,
                            name=des_name,
                            custom_data=custom_data,
                            effect_sizer=effect_sizer
                        )
                        res_dict = {
                            'designer':des_name,
                            'data_name':des_data_name,
                            'pval_func':des_pval_func_name,
                            'effect_sizer':efs_name,
                            'effect':effect,
                            'transformer':trans_name,
                            'covariates':trans_cov_name,
                        }
                        if effect:

                            mean_p = designer.calculate_some_power(effect=effect, n_jobs=-1, verbose=False).mean()
                            tests_res.append(
                                res_dict | {'test_type':'some_power', 'mean_p':mean_p}
                            )

                            mean_p = designer.calculate_power_curve(effects_list=[effect], n_jobs=-1, verbose=False, plot=False)[0][1][0].mean()
                            tests_res.append(
                                res_dict | {'test_type':'power_curve', 'mean_p':mean_p}
                            )

                        else:

                            mean_p = designer.aa_sims(n_jobs=-1, verbose=False, plot=False).mean()
                            tests_res.append(
                                res_dict | {'test_type':'aa', 'mean_p':mean_p}
                            )
                        
                        pbar.update(1)

    return tests_res

def write_exact_tests_results(path=None):
    """
    Run the exact tests and write the actual results to a csv file.

    Computes `_get_exact_tests_results()` and saves it as a csv with the same
    column layout as the expectations file, so the output can be used directly
    as a new baseline for `run_exact_tests`.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Destination csv path. Defaults to `tests_exact_expectations.csv`
        located next to this module.

    Returns
    -------
    pathlib.Path
        The path the results were written to.
    """
    _HERE = Path(__file__).parent
    if path is None:
        path = _HERE / 'tests_exact_expectations.csv'
    path = Path(path)

    tests_res = _get_exact_tests_results()
    df = pd.DataFrame(tests_res)

    df.to_csv(path, index=False)
    print(f'Wrote {len(df)} rows to {path}')

    return path

def run_exact_tests(atol=1e-9, rtol=1e-6):
    """
    Run the exact tests and compare actual results against the expected ones.

    The actual results (`tests_res`) are matched against the expected values
    from the `tests_exact_expectations.csv` file by all key columns, after
    which the `mean_p` column is compared.

    Parameters
    ----------
    atol : float
        Absolute tolerance when comparing `mean_p` (for robustness against
        floating-point noise).
    rtol : float
        Relative tolerance when comparing `mean_p`.

    Returns
    -------
    pandas.DataFrame
        Table of discrepancies (empty if no discrepancies were found).
    """

    # ---- Key columns used to match rows ----
    KEY_COLS = [
        'designer', 'data_name', 'pval_func', 'effect_sizer',
        'effect', 'transformer', 'covariates', 'test_type',
    ]

    # ---- ANSI colors for pretty terminal output ----
    class _C:
        RESET  = '\033[0m'
        BOLD   = '\033[1m'
        DIM    = '\033[2m'
        RED    = '\033[91m'
        GREEN  = '\033[92m'
        YELLOW = '\033[93m'
        CYAN   = '\033[96m'

    def _hr(char='─', width=80):
        return char * width

    print()
    print(f'{_C.BOLD}{_C.CYAN}{_hr("═")}{_C.RESET}')
    print(f'{_C.BOLD}{_C.CYAN}  EXACT TESTS: run and compare against expected results{_C.RESET}')
    print(f'{_C.BOLD}{_C.CYAN}{_hr("═")}{_C.RESET}')
    print()

    # ---- Get the actual results ----
    tests_res = _get_exact_tests_results()
    actual = pd.DataFrame(tests_res)

    # ---- Load the expected results ----
    _HERE = Path(__file__).parent
    expectations = pd.read_csv(_HERE / 'tests_exact_expectations.csv')

    # ---- Normalize keys for a correct join (NaN does not join with NaN) ----
    def _normalize(df):
        df = df.copy()
        for col in KEY_COLS:
            if col == 'effect':
                # effect is always a number — cast to float and round artifacts
                df[col] = df[col].astype(float).round(12)
            else:
                # covariates may be NaN — fill with an explicit label
                df[col] = df[col].astype(object).where(df[col].notna(), '__NA__').astype(str)
        return df

    actual_n   = _normalize(actual)
    expected_n = _normalize(expectations)

    # ---- Join actual and expected by keys ----
    merged = actual_n.merge(
        expected_n[KEY_COLS + ['mean_p']],
        on=KEY_COLS,
        how='outer',
        suffixes=('_actual', '_expected'),
        indicator=True,
    )

    n_total      = len(merged)
    only_actual  = merged[merged['_merge'] == 'left_only']
    only_expect  = merged[merged['_merge'] == 'right_only']
    matched      = merged[merged['_merge'] == 'both'].copy()

    print(f'{_C.BOLD}Summary:{_C.RESET}')
    print(f'  • actual rows   : {_C.BOLD}{len(actual)}{_C.RESET}')
    print(f'  • expected rows : {_C.BOLD}{len(expectations)}{_C.RESET}')
    print(f'  • matched       : {_C.BOLD}{len(matched)}{_C.RESET}')
    print()

    # ---- Warn about unmatched rows ----
    if len(only_actual) > 0:
        print(f'{_C.YELLOW}⚠  Rows present in actual but missing in expected: '
              f'{len(only_actual)}{_C.RESET}')
        for _, row in only_actual.iterrows():
            print(f'{_C.YELLOW}     {" | ".join(str(row[c]) for c in KEY_COLS)}{_C.RESET}')
        print()
    if len(only_expect) > 0:
        print(f'{_C.YELLOW}⚠  Rows present in expected but missing in actual: '
              f'{len(only_expect)}{_C.RESET}')
        for _, row in only_expect.iterrows():
            print(f'{_C.YELLOW}     {" | ".join(str(row[c]) for c in KEY_COLS)}{_C.RESET}')
        print()

    # ---- Compare mean_p for matched rows ----
    matched['abs_diff'] = (matched['mean_p_actual'] - matched['mean_p_expected']).abs()
    matched['pct_diff'] = np.where(
        matched['mean_p_expected'] != 0,
        matched['abs_diff'] / matched['mean_p_expected'].abs() * 100.0,
        np.where(matched['abs_diff'] == 0, 0.0, np.inf),
    )

    tolerance = atol + rtol * matched['mean_p_expected'].abs()
    matched['mismatch'] = matched['abs_diff'] > tolerance

    diffs = matched[matched['mismatch']].copy()

    # ---- Summary for mean_p ----
    print(f'{_C.BOLD}Comparison by mean_p column:{_C.RESET}')
    if len(diffs) == 0 and len(only_actual) == 0 and len(only_expect) == 0:
        print(f'{_C.GREEN}{_C.BOLD}  ✓ ALL OK — no deviations found '
              f'({len(matched)}/{len(matched)} rows matched).{_C.RESET}')
        print()
        print(f'{_C.GREEN}{_hr("═")}{_C.RESET}')
        return pd.DataFrame()

    if len(diffs) == 0:
        print(f'{_C.GREEN}  ✓ No deviations by mean_p '
              f'({len(matched)}/{len(matched)} matched rows are equal).{_C.RESET}')
    else:
        print(f'{_C.RED}{_C.BOLD}  ✗ ERROR: mean_p discrepancies found: '
              f'{len(diffs)} of {len(matched)}.{_C.RESET}')
    print()

    # ---- Detailed report for each discrepancy ----
    for n, (_, row) in enumerate(diffs.iterrows(), start=1):
        print(f'{_C.RED}{_hr("─")}{_C.RESET}')
        print(f'{_C.RED}{_C.BOLD}  ✗ DISCREPANCY #{n}{_C.RESET}')
        print(f'{_C.RED}{_hr("─")}{_C.RESET}')

        # The full row from the actual results
        print(f'  {_C.BOLD}Actual result row:{_C.RESET}')
        key_str = ', '.join(f'{c}={row[c]}' for c in KEY_COLS)
        print(f'    {_C.DIM}{key_str}{_C.RESET}')

        actual_val   = row['mean_p_actual']
        expected_val = row['mean_p_expected']
        abs_diff     = row['abs_diff']
        pct_diff     = row['pct_diff']

        print(f'    {_C.BOLD}mean_p (actual)   :{_C.RESET} {_C.RED}{actual_val:.12g}{_C.RESET}')
        print(f'    {_C.BOLD}mean_p (expected) :{_C.RESET} {_C.GREEN}{expected_val:.12g}{_C.RESET}')
        print(f'    {_C.BOLD}diff (absolute)   :{_C.RESET} {abs_diff:.12g}')
        pct_str = '∞' if np.isinf(pct_diff) else f'{pct_diff:.6f}%'
        print(f'    {_C.BOLD}diff (percent)    :{_C.RESET} {pct_str}')
        print()

    print(f'{_C.RED}{_hr("═")}{_C.RESET}')
    print(f'{_C.RED}{_C.BOLD}  TOTAL mean_p discrepancies: {len(diffs)}.{_C.RESET}')
    print(f'{_C.RED}{_hr("═")}{_C.RESET}')

    # ---- Return the table of discrepancies for further analysis ----
    report_cols = KEY_COLS + ['mean_p_actual', 'mean_p_expected', 'abs_diff', 'pct_diff']
    return diffs[report_cols].reset_index(drop=True)


if __name__ == '__main__':
    import sys

    # Usage:
    #   python -m mcab.tests                  -> compare actual vs expected (run_exact_tests)
    #   python -m mcab.tests run              -> same as above
    #   python -m mcab.tests write            -> overwrite the baseline csv next to the module
    #   python -m mcab.tests write out.csv    -> write results to a custom csv path
    args = sys.argv[1:]
    command = args[0] if args else 'run'

    if command == 'write':
        out_path = args[1] if len(args) > 1 else None
        write_exact_tests_results(out_path)
    elif command == 'run':
        run_exact_tests()
    else:
        print(f"Unknown command: {command!r}. Use 'run' or 'write [path]'.")
        sys.exit(1)
