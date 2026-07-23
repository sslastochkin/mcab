"""
mcab_benchmark.py
=================
Performance comparison:
  1. Pure Python for loop
  2. mcab  n_jobs=1
  3. mcab  n_jobs=-1  (always cold pool — each call spawns a fresh subprocess)

Three scenarios:
  A. AA test
  B. AB test  (effect applied by multiplying test group)
  C. Multiple testing  (k=4 metrics, Holm-Bonferroni)

Config:
  N_SIMS_SIMPLE = 10_000   (AA, AB)
  N_SIMS_MULTI  =  3_000   (multiple testing)
  DATA_SIZE     = 30_000
"""

# Agg backend suppresses all matplotlib windows created by mcab internals.
# Final result plot is opened via subprocess.
import matplotlib
matplotlib.use('Agg')

import subprocess
import sys
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
N_SIMS_SIMPLE = 10_000
N_SIMS_MULTI  =  3_000
N_SIMS_SCALE  = [1_000, 3_000, 5_000, 10_000, 20_000]
EFFECT        = 0.05
DATA_SIZE     = 30_000
SEED          = 42

# ══════════════════════════════════════════════════════════════════
#  DATA & MCAB SETUP
# ══════════════════════════════════════════════════════════════════
print("⚙  Initializing data and designers …")
from mcab import RandomData, DesignerIid, BenchMarker

rd   = RandomData(seed=SEED)
pval = lambda t, c: stats.ttest_ind(t, c, equal_var=False).pvalue

data_main = rd.exponential_data(size=DATA_SIZE, scale=1_000_000)
designer  = DesignerIid(
    target       = data_main,
    alpha        = 0.05,
    effect_sizer = 'percent',
    pval_func    = pval,
)

scales       = [1_000_000, 500_000, 2_000_000, 3_000_000]
data_multi   = [rd.exponential_data(size=DATA_SIZE, scale=s) for s in scales]
designers_mt = [
    DesignerIid(target=d, alpha=0.05, effect_sizer='percent', pval_func=pval)
    for d in data_multi
]
bm = BenchMarker(designers_mt)
print("   done.\n")

# ══════════════════════════════════════════════════════════════════
#  FOR-LOOP IMPLEMENTATIONS
#
#  Split strategy: one rng.choice picks test indices;
#  the complement (boolean mask) becomes control.
# ══════════════════════════════════════════════════════════════════

def forloop_aa(data, n_sims, seed=SEED):
    """AA test: single rng.choice split, t-test, collect p-values."""
    rng   = np.random.default_rng(seed)
    n     = len(data)
    half  = n // 2
    pvals = np.empty(n_sims)
    mask  = np.zeros(n, dtype=bool)
    for i in range(n_sims):
        idx       = rng.choice(n, size=half, replace=False)
        mask[:]   = False
        mask[idx] = True
        pvals[i]  = pval(data[mask], data[~mask])
    return pvals


def forloop_ab(data, n_sims, effect, seed=SEED):
    """AB test: single rng.choice split, effect applied to test group."""
    rng   = np.random.default_rng(seed)
    n     = len(data)
    half  = n // 2
    pvals = np.empty(n_sims)
    mask  = np.zeros(n, dtype=bool)
    for i in range(n_sims):
        idx       = rng.choice(n, size=half, replace=False)
        mask[:]   = False
        mask[idx] = True
        pvals[i]  = pval(data[mask] * (1.0 + effect), data[~mask])
    return pvals


def _holm_any_reject(pvals_arr, alpha=0.05):
    """
    Fast inline Holm-Bonferroni step-down.
    Returns True if at least one H0 is rejected.
    Under Holm, at least one rejection iff min(p) <= alpha/k.
    """
    return np.min(pvals_arr) <= alpha / len(pvals_arr)


def forloop_multi(data_list, n_sims, seed=SEED):
    """
    Multiple testing: single rng.choice split per dataset,
    fast inline Holm-Bonferroni correction.
    """
    rng    = np.random.default_rng(seed)
    k      = len(data_list)
    n_min  = min(len(d) for d in data_list)
    half   = n_min // 2
    masks  = [np.zeros(n_min, dtype=bool) for _ in data_list]
    pvals_arr = np.empty(k)
    reject = np.zeros(n_sims, dtype=bool)
    for i in range(n_sims):
        for j, (d, m) in enumerate(zip(data_list, masks)):
            idx     = rng.choice(n_min, size=half, replace=False)
            m[:]    = False
            m[idx]  = True
            pvals_arr[j] = pval(d[m], d[~m])
        reject[i] = _holm_any_reject(pvals_arr)
    return reject


# ══════════════════════════════════════════════════════════════════
#  TIMING UTILITIES
# ══════════════════════════════════════════════════════════════════

def timed(func, *args, **kwargs):
    """Time an in-process function call."""
    t0     = time.perf_counter()
    result = func(*args, **kwargs)
    plt.close('all')    # clean up any figures mcab may have created
    return result, time.perf_counter() - t0


def timed_subprocess(scenario, n_sims):
    """
    Run _bench_worker.py in a fresh subprocess — guarantees a truly
    cold joblib/loky worker pool for every n_jobs=-1 measurement.
    Returns elapsed seconds (float) as reported by the worker itself
    (excludes Python startup time, includes loky pool spawn).
    """
    cmd = [
        sys.executable, '_bench_worker.py',
        scenario, str(n_sims), str(DATA_SIZE), str(EFFECT), str(SEED)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(proc.stdout.strip())


# ══════════════════════════════════════════════════════════════════
#  MAIN BENCHMARK
# ══════════════════════════════════════════════════════════════════
print(f"🔬 Running benchmark  (data_size={DATA_SIZE:,}) …\n")
results = {}

# ── AA ──────────────────────────────────────────────────────────
print(f"  [1/3] AA test  (n_sims={N_SIMS_SIMPLE:,})")
results['AA Test'] = {}
_, results['AA Test']['For Loop']       = timed(forloop_aa, data_main, N_SIMS_SIMPLE)
_, results['AA Test']['mcab n_jobs=1']  = timed(
    designer.aa_sims, n_sims=N_SIMS_SIMPLE, verbose=False, plot=False, n_jobs=1)
results['AA Test']['mcab n_jobs=-1']    = timed_subprocess('aa', N_SIMS_SIMPLE)
for k, v in results['AA Test'].items():
    print(f"     {k:<18}: {v:.2f}s")

# ── AB ──────────────────────────────────────────────────────────
print(f"\n  [2/3] AB test  (n_sims={N_SIMS_SIMPLE:,}, effect={EFFECT:.0%})")
results['AB Test'] = {}
_, results['AB Test']['For Loop']       = timed(forloop_ab, data_main, N_SIMS_SIMPLE, EFFECT)
_, results['AB Test']['mcab n_jobs=1']  = timed(
    designer.calculate_some_power, effect=EFFECT,
    n_sims=N_SIMS_SIMPLE, verbose=False, n_jobs=1)
results['AB Test']['mcab n_jobs=-1']    = timed_subprocess('ab', N_SIMS_SIMPLE)
for k, v in results['AB Test'].items():
    print(f"     {k:<18}: {v:.2f}s")

# ── Multiple Testing ─────────────────────────────────────────────
print(f"\n  [3/3] Multiple testing  (n_sims={N_SIMS_MULTI:,}, k={len(data_multi)}, Holm-Bonferroni)")
results['Multiple\nTesting'] = {}
_, results['Multiple\nTesting']['For Loop']      = timed(forloop_multi, data_multi, N_SIMS_MULTI)
_, results['Multiple\nTesting']['mcab n_jobs=1'] = timed(
    bm.multi_test_1_type_error,
    multi_corrections='holm_bonferroni', n_sims=N_SIMS_MULTI, verbose=False, n_jobs=1)
results['Multiple\nTesting']['mcab n_jobs=-1']   = timed_subprocess('multi', N_SIMS_MULTI)
for k, v in results['Multiple\nTesting'].items():
    print(f"     {k:<18}: {v:.2f}s")

# ══════════════════════════════════════════════════════════════════
#  SCALING BENCHMARK  (AA, variable n_sims)
# ══════════════════════════════════════════════════════════════════
print("\n📈 Scaling benchmark (AA test) …")
scale_times = {'For Loop': [], 'mcab n_jobs=1': [], 'mcab n_jobs=-1': []}

for ns in N_SIMS_SCALE:
    print(f"   n_sims={ns:>6,} … ", end='', flush=True)
    _, t_fl = timed(forloop_aa, data_main, ns)
    _, t_1  = timed(designer.aa_sims, n_sims=ns, verbose=False, plot=False, n_jobs=1)
    t_p     = timed_subprocess('aa', ns)
    scale_times['For Loop'].append(t_fl)
    scale_times['mcab n_jobs=1'].append(t_1)
    scale_times['mcab n_jobs=-1'].append(t_p)
    print(f"for_loop={t_fl:.1f}s  n_jobs=1={t_1:.1f}s  n_jobs=-1={t_p:.1f}s")

print()

# ══════════════════════════════════════════════════════════════════
#  PLOT
# ══════════════════════════════════════════════════════════════════
print("🎨 Building figure …")

plt.rcParams.update({
    'font.family'      : 'sans-serif',
    'font.sans-serif'  : ['Inter', 'Helvetica Neue', 'Arial'],
    'axes.facecolor'   : '#0f1117',
    'figure.facecolor' : '#090b15',
    'axes.edgecolor'   : '#252845',
    'axes.labelcolor'  : '#c8cedf',
    'xtick.color'      : '#c8cedf',
    'ytick.color'      : '#8a8fb0',
    'grid.color'       : '#1c2040',
    'grid.linestyle'   : '--',
    'text.color'       : '#c8cedf',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

CLRS = {
    'For Loop'       : '#4a5278',
    'mcab n_jobs=1'  : '#00c6a7',
    'mcab n_jobs=-1' : '#9f7aea',
}
METHODS   = list(CLRS.keys())
SCENARIOS = list(results.keys())

fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    f'MCAB Performance Benchmark  —  '
    f'AA/AB n_sims={N_SIMS_SIMPLE:,}  |  '
    f'Multi n_sims={N_SIMS_MULTI:,}  |  '
    f'data_size={DATA_SIZE:,}  |  n_jobs=-1: cold pool (subprocess)',
    fontsize=13, fontweight='bold', color='white', y=0.98
)
gs = gridspec.GridSpec(2, 2, figure=fig,
                       hspace=0.48, wspace=0.38,
                       left=0.07, right=0.97, top=0.92, bottom=0.08)

# ─── Panel 1: absolute time (full-width top row) ─────────────────
ax1 = fig.add_subplot(gs[0, :])

n_sc  = len(SCENARIOS)
width = 0.22
x     = np.arange(n_sc)
max_t = max(results[sc][m] for sc in SCENARIOS for m in METHODS)

for mi, method in enumerate(METHODS):
    offsets = (mi - 1) * (width + 0.02)
    vals    = [results[sc][method] for sc in SCENARIOS]
    bars    = ax1.bar(x + offsets, vals, width,
                      color=CLRS[method], alpha=0.88,
                      label=method, zorder=3)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max_t * 0.012,
                 f'{v:.1f}s',
                 ha='center', va='bottom',
                 fontsize=8.5, color=CLRS[method], fontweight='700')

ax1.set_xticks(x)
ax1.set_xticklabels(SCENARIOS, fontsize=12, fontweight='700', color='white')
ax1.set_ylabel('Elapsed Time (seconds)', fontsize=11)
ax1.set_title('Absolute Execution Time per Scenario & Method', fontsize=13,
              fontweight='bold', color='white', pad=10)
ax1.grid(axis='y', zorder=0)
ax1.set_axisbelow(True)
# Extra headroom so bar labels don't collide with the top or legend
ax1.set_ylim(0, max_t * 1.28)
# Legend outside the plot area, below the x-axis
ax1.legend(framealpha=0.15, facecolor='#1a1d2e', edgecolor='#252845',
           fontsize=10, loc='upper center',
           bbox_to_anchor=(0.5, -0.10), ncol=3)

# ─── Panel 2: speedup vs for loop ────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])

sp1 = [results[sc]['For Loop'] / results[sc]['mcab n_jobs=1']  for sc in SCENARIOS]
spp = [results[sc]['For Loop'] / results[sc]['mcab n_jobs=-1'] for sc in SCENARIOS]

bw = 0.32
ax2.bar(x - bw/2, sp1, bw, color=CLRS['mcab n_jobs=1'],  alpha=0.88, label='mcab n_jobs=1',  zorder=3)
ax2.bar(x + bw/2, spp, bw, color=CLRS['mcab n_jobs=-1'], alpha=0.88, label='mcab n_jobs=-1', zorder=3)

for xi, (s1, sp) in enumerate(zip(sp1, spp)):
    ax2.text(xi - bw/2, s1 + 0.04, f'{s1:.1f}×', ha='center', va='bottom',
             fontsize=9.5, fontweight='800', color=CLRS['mcab n_jobs=1'])
    ax2.text(xi + bw/2, sp + 0.04, f'{sp:.1f}×', ha='center', va='bottom',
             fontsize=9.5, fontweight='800', color=CLRS['mcab n_jobs=-1'])

ax2.axhline(1.0, color='#4a5278', linewidth=1.4, linestyle='--', label='Baseline (for loop)')
ax2.set_xticks(x)
ax2.set_xticklabels(SCENARIOS, fontsize=10, color='white')
ax2.set_ylabel('Speedup  (× faster than for loop)', fontsize=10)
ax2.set_title('Speedup vs. For Loop', fontsize=12, fontweight='bold', color='white', pad=8)
ax2.grid(axis='y', zorder=0)
ax2.set_axisbelow(True)
ax2.set_ylim(0, max(max(sp1), max(spp)) * 1.28)
ax2.legend(framealpha=0.15, facecolor='#1a1d2e', edgecolor='#252845', fontsize=9,
           loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)

# ─── Panel 3: scaling — AA test ───────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])

for method in METHODS:
    ax3.plot(N_SIMS_SCALE, scale_times[method],
             marker='o', markersize=6,
             color=CLRS[method], linewidth=2.4,
             label=method, zorder=3)
    ax3.fill_between(N_SIMS_SCALE, 0, scale_times[method],
                     color=CLRS[method], alpha=0.07)

ax3.set_xlabel('Number of Simulations', fontsize=10)
ax3.set_ylabel('Elapsed Time (seconds)', fontsize=10)
ax3.set_title('Scaling: AA Test  (time vs. n_sims)', fontsize=12,
              fontweight='bold', color='white', pad=8)
ax3.grid(zorder=0)
ax3.set_axisbelow(True)
ax3.legend(framealpha=0.15, facecolor='#1a1d2e', edgecolor='#252845', fontsize=9)
ax3.set_xlim(left=0)
ax3.set_ylim(bottom=0)

fig.text(0.5, 0.012,
         f'Effect: {EFFECT:.0%}  |  {len(data_multi)} metrics for multiple testing  |  '
         f'Data: Exponential(scale=1M)  |  Holm-Bonferroni correction  |  '
         f'n_jobs=-1 measured in isolated subprocess (cold loky pool)',
         ha='center', fontsize=8.0, color='#4a5278')

out = 'mcab_benchmark.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"   saved → {out}")

# Open the saved image (macOS: open; Linux: xdg-open)
try:
    subprocess.Popen(['open', out])
except FileNotFoundError:
    try:
        subprocess.Popen(['xdg-open', out])
    except FileNotFoundError:
        print("   (open mcab_benchmark.png manually)")

print("\n✅ Done.")
