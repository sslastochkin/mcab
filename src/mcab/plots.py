"""
Visualization functions for A/B testing analysis.
"""
from typing import Tuple, List, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mc
import seaborn as sns
from scipy import stats


def lighten_color(hex_color: str, amount: float = 0.5) -> tuple:
    """
    Lighten a color by a given amount (0..1).
    
    Parameters
    ----------
    hex_color : str
        Color in hex format.
    amount : float, default=0.5
        Lightening amount from 0 to 1.
    
    Returns
    -------
    tuple
        RGB tuple of lightened color.
    """
    rgb = mc.to_rgb(hex_color)
    return tuple(c + (1 - c) * amount for c in rgb)


def _get_aa_plot_color(mean_p_left: float, mean_p_right: float, 
                       mean_p_ft_left: float, ks_p_value: float) -> str:
    """
    Determine plot color based on AA test diagnostics.
    
    Parameters
    ----------
    mean_p_left : float
        Left CI bound for mean p-value.
    mean_p_right : float
        Right CI bound for mean p-value.
    mean_p_ft_left : float
        Left CI bound for first type error.
    ks_p_value : float
        Kolmogorov-Smirnov test p-value.
    
    Returns
    -------
    str
        Color code (green/red/orange/purple).
    """
    good_color = '#2ca02c'
    bad_color = '#d62728'
    neutral_color = '#ff7f0e'
    other_color = '#9467bd'
    
    mean_pvalue_check = mean_p_left <= 0.5 and mean_p_right >= 0.5
    first_error_check = mean_p_ft_left <= 0.05
    ks_check = ks_p_value >= 0.05
    
    result_is_good = all([mean_pvalue_check, first_error_check, ks_check])
    result_is_bad = (not mean_pvalue_check) and (not first_error_check)
    result_is_neutral = not result_is_good and not result_is_bad
    
    if result_is_good:
        return good_color
    elif result_is_bad:
        return bad_color
    elif result_is_neutral:
        return neutral_color
    else:
        return other_color


def plot_aa_histogram(pvals: np.ndarray, color: str, ax):
    """
    Plot histogram of p-values.
    
    Parameters
    ----------
    pvals : ndarray
        Array of p-values.
    color : str
        Bar color.
    ax : Axes
        Matplotlib axis.
    """
    sns.histplot(pvals, stat='density', ax=ax, color=color)
    ax.plot([0, 1], [1, 1], 'k--')
    ax.set(xlabel='p-value', ylabel='Density')
    ax.set_title('Histogram')


def plot_aa_qqplot(pvals: np.ndarray, color: str, ax):
    """
    Plot QQ-plot for p-values against uniform distribution.
    
    Parameters
    ----------
    pvals : ndarray
        Array of p-values.
    color : str
        Line color.
    ax : Axes
        Matplotlib axis.
    """
    stats.probplot(pvals, dist='uniform', plot=ax, sparams=(0, 1))
    ax.get_lines()[1].set_linestyle('--')
    ax.get_lines()[1].set_color('black')
    ax.get_lines()[0].set_color(color)
    ax.set_title('QQ-plot')


def plot_aa_ecdf(pvals: np.ndarray, color: str, ax):
    """
    Plot empirical CDF vs theoretical uniform CDF.
    
    Parameters
    ----------
    pvals : ndarray
        Array of p-values.
    color : str
        Line color.
    ax : Axes
        Matplotlib axis.
    """
    pvals_sorted = np.sort(pvals)
    n = len(pvals_sorted)
    ecdf = np.arange(1, n+1) / n
    
    x_uniform = np.linspace(0, 1, 200)
    cdf_uniform = x_uniform
    
    ax.step(pvals_sorted, ecdf, where='post', label='Emperical CDF', color=color)
    ax.plot(x_uniform, cdf_uniform, 'k--', label='Theoretical CDF $U[0,1]$')
    ax.set_title('ECDF vs CDF')
    ax.set_xlabel('p-value')
    ax.set_ylabel('F(p-value)')
    ax.legend(loc="lower right")


def plot_aa_convergence(pvals: np.ndarray, alpha: float, color: str,
                        mean_p: float, mean_p_left: float, mean_p_right: float,
                        mean_p_ft: float, mean_p_ft_left: float, mean_p_ft_right: float,
                        ks_statistic: float, ks_p_value: float, cumulative_mean: np.ndarray, ax):
    """
    Plot convergence of Type I error rate.
    
    Parameters
    ----------
    pvals : ndarray
        Array of p-values.
    alpha : float
        Significance level.
    color : str
        Line color.
    mean_p : float
        Mean p-value.
    mean_p_left : float
        Left CI bound for mean p-value.
    mean_p_right : float
        Right CI bound for mean p-value.
    mean_p_ft : float
        First type error rate.
    mean_p_ft_left : float
        Left CI bound for first type error.
    mean_p_ft_right : float
        Right CI bound for first type error.
    ks_statistic : float
        Kolmogorov-Smirnov statistic.
    ks_p_value : float
        Kolmogorov-Smirnov p-value.
    cumulative_mean : ndarray
        Pre-calculated cumulative mean of errors.
    ax : Axes
        Matplotlib axis.
    """
    errors = (pvals < alpha).astype(int)
    
    ax.plot(
        np.arange(1, len(pvals) + 1),
        cumulative_mean,
        label=f"I type error: {errors.mean():.4f}",
        linewidth=1.5,
        alpha=0.85,
        color=color
    )
    
    ax.axhline(y=alpha, color="red", linestyle="--", linewidth=1.5,
               label=f"α = {alpha}", alpha=0.7)
    ax.set_xlabel("Simulation number", fontsize=11)
    ax.set_ylabel("Cumulative I type error rate", fontsize=11)
    ax.set_title("I Type Error Convergence by sim", fontsize=12)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Determine check signs
    good_sign = '[OK]'
    bad_sign = '[X]'
    mean_pvalue_check = mean_p_left <= 0.5 and mean_p_right >= 0.5
    first_error_check = mean_p_ft_left <= 0.05
    ks_check = ks_p_value >= 0.05
    
    mean_pvalue_sign = good_sign if mean_pvalue_check else bad_sign
    first_error_sign = good_sign if first_error_check else bad_sign
    ks_sign = good_sign if ks_check else bad_sign
    
    # Summary text
    ax.text(
        0.36, 0.35, "Summary",
        transform=ax.transAxes,
        fontsize=10, fontweight='bold',
        verticalalignment='top'
    )
    
    ax.text(
        .36, 0.30,
        f"{mean_pvalue_sign} Mean pvalue: {mean_p:.3f} [{mean_p_left:.3f}, {mean_p_right:.3f}]\n"
        f"{first_error_sign} First type error: {mean_p_ft:.3f} [{mean_p_ft_left:.3f}, {mean_p_ft_right:.3f}]\n"
        f"{ks_sign} Kolmogorov-Smirnov:\n - stat={ks_statistic:.3f}\n - pvalue={ks_p_value:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle="round", fc="w", alpha=0.5),
    )


def plot_aa_pvalues(pvals: np.ndarray, alpha: float, mean_ci_func, first_type_error_ci_func, cumulative_mean: np.ndarray, ax=None, figsize=None):
    """
    AA-simulations plot for p-values diagnostics.
    
    Creates 4 diagnostic plots:
    1. Histogram of p-values
    2. QQ-plot against uniform distribution
    3. Empirical CDF vs theoretical CDF
    4. Type I error convergence
    
    Parameters
    ----------
    pvals : ndarray
        Array of p-values from AA simulations.
    alpha : float
        Significance level.
    mean_ci_func : callable
        Function to calculate mean and CI.
    first_type_error_ci_func : callable
        Function to calculate first type error CI.
    cumulative_mean : ndarray
        Pre-calculated cumulative mean of errors.
    ax : array-like of Axes, optional
        Array of 4 axes for plots.
    figsize : tuple, optional
        Figure size (width, height). Default is (30, 6).
    """
    if ax is None:
        if figsize is None:
            figsize = (30, 6)
        fig, ax = plt.subplots(1, 4, figsize=figsize)
    
    # Calculate statistics
    mean_p, mean_p_left, mean_p_right = mean_ci_func(pvals, alpha=alpha)
    mean_p_ft, mean_p_ft_left, mean_p_ft_right = first_type_error_ci_func(pvals, alpha=alpha)
    ks_statistic, ks_p_value = stats.kstest(pvals, 'uniform', args=(0, 1))
    
    # Determine color
    color = _get_aa_plot_color(mean_p_left, mean_p_right, mean_p_ft_left, ks_p_value)
    
    # Create plots
    plot_aa_histogram(pvals, color, ax[0])
    plot_aa_qqplot(pvals, color, ax[1])
    plot_aa_ecdf(pvals, color, ax[2])
    plot_aa_convergence(pvals, alpha, color, mean_p, mean_p_left, mean_p_right,
                       mean_p_ft, mean_p_ft_left, mean_p_ft_right,
                       ks_statistic, ks_p_value, cumulative_mean, ax[3])
    
    plt.show()


def plot_power_cum_means(
    effects_list: List[float],
    cummeans: List[np.ndarray],
    target_power: Optional[float] = None,
    ax=None,
    figsize=None
):
    """
    Plot cumulative means of power for different effects.
    
    Parameters
    ----------
    effects_list : list of float
        List of effect sizes.
    cummeans : list of ndarray
        List of cumulative means for each effect.
    target_power : float, optional
        Target power for horizontal line.
    ax : Axes, optional
        Matplotlib axis.
    figsize : tuple, optional
        Figure size (width, height). Default is (20, 8).
    """
    if ax is None:
        if figsize is None:
            figsize = (20, 8)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i, effect in enumerate(effects_list):
        ax.plot(
            np.arange(1, len(cummeans[i]) + 1),
            cummeans[i],
            label=f"effect={effect:.4f}",
            linewidth=1.5,
            alpha=0.85
        )
    ax.set_xlabel('Simulation number', fontsize=20)
    ax.set_ylabel('Cumulative average power', fontsize=20)
    ax.set_title('Power Convergence by sim', fontsize=24, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=target_power, color='r', linestyle='--', linewidth=3.5)
    ax.legend(loc='lower right', fontsize=14, framealpha=0.9)
    ax.tick_params(labelsize=18)

    plt.show()


def plot_power_curve_data(
    effects_list: List[float],
    power_curves: List[tuple],
    cummeans: List[List[np.ndarray]],
    target_power: Optional[float] = None,
    annotate: bool = False,
    figsize=None
):
    """
    Plot power curves.
    
    Parameters
    ----------
    effects_list : list of float
        List of effect sizes.
    power_curves : list of tuple
        List of tuples (name, power_array).
    cummeans : list of list of ndarray
        Cumulative means for each function and effect.
    target_power : float, optional
        Target power level.
    annotate : bool, default=False
        Whether to annotate points with values.
    figsize : tuple, optional
        Figure size (width, height). Default is (16, 16) for single curve or (16, 8) for multiple.
    """
    if figsize is None:
        if len(power_curves) == 1:
            figsize = (16, 16)
        else:
            figsize = (16, 8)
    
    if len(power_curves) == 1:
        fig, ax = plt.subplots(2, 1, figsize=figsize)
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax = [ax]
    
    for fname, power_curve in power_curves:
        ax[0].plot(
            effects_list,
            power_curve,
            label=fname,
            linewidth=6.,
            marker='s',
            markersize=14
        )
        if annotate:
            for x, y in zip(effects_list, power_curve):
                ax[0].annotate(f"{y:.4f}", (x, y+0.02), fontsize=18)
    
    ax[0].set_xlabel('Effect size', fontsize=20)
    ax[0].set_ylabel('Power', fontsize=20)
    ax[0].set_title('Power curve', fontsize=24, fontweight="bold")
    ax[0].axhline(y=target_power, color='r', linestyle='--', linewidth=3.5)
    ax[0].grid(True, alpha=0.3)
    ax[0].tick_params(labelsize=18)
    ax[0].legend(loc='lower right', fontsize=14, framealpha=0.9)

    if len(power_curves) == 1:
        plot_power_cum_means(
            effects_list=effects_list,
            cummeans=cummeans[0],
            target_power=target_power,
            ax=ax[1]
        )

    plt.show()


def plot_comparison_bars(
    baseline: float,
    values: Union[np.array, List[float]],
    labels: List[str] = None,
    lesser_is_good: bool = True,
    sort: bool = True,
    ax=None,
    title: Optional[str] = None,
    ylabel: str = 'Values',
    figsize: Optional[Tuple[int, int]] = None
):
    """
    Plot bars for comparison with baseline.
    
    Parameters
    ----------
    baseline : float
        Baseline value - will be first gray bar labeled "Baseline".
    values : array-like
        List of values for comparison columns.
    labels : list of str, optional
        Labels for comparison columns.
    lesser_is_good : bool, default=True
        If True: less than baseline = green, more = red; otherwise reversed.
    sort : bool, default=True
        If True: sort comparison columns with best right after baseline.
    ax : Axes, optional
        Matplotlib axis.
    title : str, optional
        Plot title.
    ylabel : str, default='Values'
        Y-axis label.
    
    Returns
    -------
    ax : Axes
        Matplotlib axis with plot.
    res : list of tuple
        List of tuples (label, value, pct).
    """
    labels = labels or list(range(len(values)))
    assert len(labels) == len(values), 'labels and values must have the same length'
    pairs = list(zip(labels, values))

    if sort:
        pairs = sorted(pairs, key=lambda x: x[1], reverse=not lesser_is_good)

    sorted_labels = [p[0] for p in pairs]
    sorted_values = [p[1] for p in pairs]

    all_labels = ['Baseline'] + sorted_labels
    all_values = [baseline] + sorted_values
    n = len(all_values)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (2 + 1.4 * n, 3.5))

    bar_width = 0.6
    x_pos = list(range(n))

    GREY = '#808080'
    GREEN = '#43a047'
    RED = '#e53935'

    max_val = max(v for v in all_values if v is not None) or 1
    offset_abs = max_val * 0.03

    res = []

    for i, (label, val) in enumerate(zip(all_labels, all_values)):
        diff = val - baseline
        pct = 0 if i == 0 else None

        if i == 0:
            bar_color = GREY
        elif diff == 0:
            bar_color = GREY
        elif lesser_is_good:
            bar_color = GREEN if diff < 0 else RED
        else:
            bar_color = GREEN if diff > 0 else RED

        light_color = lighten_color(bar_color, amount=0.5)

        if i == 0:
            ax.bar(x_pos[i], val, width=bar_width, color=bar_color, zorder=2)
        else:
            min_val = min(baseline, val)
            rect_main = Rectangle(
                (x_pos[i] - bar_width / 2, 0), bar_width, min_val,
                facecolor=bar_color, edgecolor='none', zorder=2
            )
            ax.add_patch(rect_main)

            if abs(diff) > 0:
                rect_light = Rectangle(
                    (x_pos[i] - bar_width / 2, min_val), bar_width, abs(diff),
                    facecolor=light_color, edgecolor='none', zorder=3
                )
                ax.add_patch(rect_light)

            if baseline != 0 and abs(diff) > 0:
                pct = (diff / baseline) * 100
                pct_str = f'{pct:+.1f}%'
                text_y = min_val + abs(diff) / 2
                ax.text(
                    x_pos[i], text_y, pct_str,
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', color='black'
                )

        top_y = max(baseline, val) if i > 0 else val
        ax.text(
            x_pos[i], top_y + offset_abs, f'{val:,.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )

        res.append((label, val, pct))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_labels, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_ylim(0, max_val * 1.2 + offset_abs * 4)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    return ax, res


def plot_aa_benchmark(
    all_stats: List[dict],
    labels: List[str],
    alpha: float,
    plot_hists: bool = True,
    all_pvalues: Optional[List[np.ndarray]] = None,
    figsize: Optional[Tuple[int, int]] = None
):
    """
    Visualization of AA-benchmark results for comparing designers.
    
    Parameters
    ----------
    all_stats : list of dict
        Statistics for each designer.
    labels : list of str
        Designer names.
    alpha : float
        Significance level.
    plot_hists : bool, default=True
        Whether to show p-value histograms.
    all_pvalues : list of ndarray, optional
        P-value arrays for histograms.
    figsize : tuple, optional
        Figure size (width, height).
    """
    n_designers = len(labels)

    # One stable color per designer, reused across every subplot so that a
    # designer keeps the same color in the bars, the convergence lines and the
    # ECDF curves.
    palette = sns.color_palette('tab10', n_designers)
    y_pos = np.arange(n_designers)

    # 2x2 grid:
    #   (0,0) I Type Error Comparison      (0,1) I Type Error Convergence
    #   (1,0) Mean p-value Comparison      (1,1) p-value ECDF
    fig, axes = plt.subplots(2, 2, figsize=figsize or (14, 9))
    ax1 = axes[0, 0]  # I Type Error Comparison
    ax3 = axes[0, 1]  # I Type Error Convergence
    ax2 = axes[1, 0]  # Mean p-value Comparison
    ax4 = axes[1, 1]  # p-value ECDF

    # Plot 1: Type I error (horizontal bars)
    ft_vals = np.array([s['mean_p_ft'] for s in all_stats])
    ft_lo = np.array([s['mean_p_ft_left'] for s in all_stats])
    ft_hi = np.array([s['mean_p_ft_right'] for s in all_stats])
    ax1.barh(y_pos, ft_vals, color=palette, alpha=0.7)
    ax1.errorbar(
        ft_vals, y_pos,
        xerr=[ft_vals - ft_lo, ft_hi - ft_vals],
        fmt='o', color='black', capsize=5, linestyle='none'
    )
    # Anchor the value label to the RIGHT end of the confidence interval (not to
    # the bar end) so it never collides with the error-bar whiskers, which on
    # this plot often extend well past the bar.
    ax1_xmax = max(ft_hi.max(), alpha)
    ft_offset = 0.03 * ax1_xmax
    for i, v in enumerate(ft_vals):
        ax1.text(ft_hi[i] + ft_offset, y_pos[i], f"{v:.4f}",
                 va='center', ha='left', fontsize=10, fontweight='bold')
    ax1.axvline(x=alpha, color='red', linestyle='--', linewidth=2, label=f'α = {alpha}')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('I Type Error Rate', fontsize=12)
    ax1.set_title('I Type Error Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(axis='x', alpha=0.3)
    # Leave room on the right so the labels never overflow the axes.
    ax1.set_xlim(0, ax1_xmax * 1.25 + ft_offset)
    ax1.invert_yaxis()

    # Plot 2: Mean p-value (horizontal bars)
    mp_vals = np.array([s['mean_p'] for s in all_stats])
    mp_lo = np.array([s['mean_p_left'] for s in all_stats])
    mp_hi = np.array([s['mean_p_right'] for s in all_stats])
    ax2.barh(y_pos, mp_vals, color=palette, alpha=0.7)
    ax2.errorbar(
        mp_vals, y_pos,
        xerr=[mp_vals - mp_lo, mp_hi - mp_vals],
        fmt='o', color='black', capsize=5, linestyle='none'
    )
    # Anchor the label to the right CI bound (consistent with the Type I error
    # plot) and keep it vertically centered on the bar.
    ax2_xmax = max(mp_hi.max(), 0.5)
    mp_offset = 0.03 * ax2_xmax
    for i, v in enumerate(mp_vals):
        ax2.text(mp_hi[i] + mp_offset, y_pos[i], f"{v:.4f}",
                 va='center', ha='left', fontsize=10, fontweight='bold')
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Expected = 0.5')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('Mean p-value', fontsize=12)
    ax2.set_title('Mean p-value Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(axis='x', alpha=0.3)
    # Leave room on the right so the labels never overflow the axes.
    ax2.set_xlim(0, ax2_xmax * 1.2 + mp_offset)
    ax2.invert_yaxis()

    # Plot 3: Convergence (line plot)
    for i, stats_dict in enumerate(all_stats):
        ax3.plot(
            np.arange(1, len(stats_dict['_1_type_error_cummeans']) + 1),
            stats_dict['_1_type_error_cummeans'],
            label=f"{labels[i]} (KS p={stats_dict['ks_p_value']:.3f}, FTE={stats_dict['mean_p_ft']:.3f})",
            color=palette[i],
            linewidth=2,
            alpha=0.85,
        )
    ax3.axhline(y=alpha, color='red', linestyle='--', linewidth=2, label=f'α = {alpha}')
    ax3.set_xlabel('Simulation number', fontsize=12)
    ax3.set_ylabel('Cumulative mean p-value', fontsize=12)
    ax3.set_title('I Type Error Convergence', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: p-value ECDF (all designers on a single axis)
    if all_pvalues is not None:
        for i, pvalues in enumerate(all_pvalues):
            sns.ecdfplot(pvalues, ax=ax4, color=palette[i], linewidth=2, label=labels[i])
        # Under H0 p-values are U[0,1], whose ECDF is the diagonal y = x.
        ax4.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2,
                 label='Uniform U[0,1]')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
    ax4.set_xlabel('p-value', fontsize=12)
    ax4.set_ylabel('ECDF', fontsize=12)
    ax4.set_title('p-value ECDF', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_ab_benchmark(
    all_bar_stats: List[dict],
    all_power_curves: List[np.ndarray],
    labels: List[str],
    beta: float,
    bar_effect: Optional[float] = None,
    power_curve_effects: Optional[List[float]] = None,
    figsize: Optional[Tuple[int, int]] = None
):
    """
    Visualization of AB-benchmark results for comparing designer power.
    
    Parameters
    ----------
    all_bar_stats : list of dict
        Power statistics for a single effect.
    all_power_curves : list of ndarray
        Power curves for different effects.
    labels : list of str
        Designer names.
    beta : float
        Type II error probability.
    bar_effect : float, optional
        Effect size for bar chart.
    power_curve_effects : list of float, optional
        List of effect sizes for power curves.
    figsize : tuple, optional
        Figure size (width, height).
    """
    required_power = 1 - beta
    n_designers = len(labels)
    
    # Determine number of plots
    n_plots = 0
    if bar_effect:
        n_plots += 2  # bar chart + convergence
    if power_curve_effects:
        n_plots += 1  # power curves
    
    if n_plots == 0:
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize or (10 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    # One stable color per designer, shared across every subplot (and matching
    # the palette used in plot_aa_benchmark).
    palette = sns.color_palette('tab10', n_designers)
    plot_idx = 0
    
    # Plot 1: Power bars (horizontal) - only if bar_effect
    if bar_effect and all_bar_stats:
        ax = axes[plot_idx]
        plot_idx += 1
        
        y_pos = np.arange(n_designers)
        bp_vals = np.array([s['mean_p'] for s in all_bar_stats])
        bp_lo = np.array([s['mean_p_left'] for s in all_bar_stats])
        bp_hi = np.array([s['mean_p_right'] for s in all_bar_stats])
        ax.barh(y_pos, bp_vals, color=palette, alpha=0.7)
        ax.errorbar(
            bp_vals, y_pos,
            xerr=[bp_vals - bp_lo, bp_hi - bp_vals],
            fmt='o', color='black', capsize=5, linestyle='none'
        )
        # Anchor the value label to the right CI bound (consistent with
        # plot_aa_benchmark) so it never collides with the error-bar whiskers.
        bp_xmax = max(bp_hi.max(), required_power)
        bp_offset = 0.03 * bp_xmax
        for i, v in enumerate(bp_vals):
            ax.text(bp_hi[i] + bp_offset, y_pos[i], f"{v:.4f}",
                    va='center', ha='left', fontsize=10, fontweight='bold')
        ax.axvline(x=required_power, color='red', linestyle='--', linewidth=2,
                  label=f'Required Power = {required_power:.2f}')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Power', fontsize=12)
        ax.set_title(f'Power Comparison (effect={bar_effect:.4f})', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        # Leave room on the right so the labels never overflow the axes.
        ax.set_xlim(0, bp_xmax * 1.2 + bp_offset)
        ax.invert_yaxis()
    
    # Plot 2: Convergence - only if bar_effect
    if bar_effect and all_bar_stats:
        ax = axes[plot_idx]
        plot_idx += 1
        
        for i, stats_dict in enumerate(all_bar_stats):
            ax.plot(
                np.arange(1, len(stats_dict['cummeans']) + 1),
                stats_dict['cummeans'],
                label=f"{labels[i]} (Power={stats_dict['mean_p']:.3f})",
                color=palette[i],
                linewidth=2,
                alpha=0.85
            )
        
        ax.axhline(y=required_power, color='red', linestyle='--', linewidth=2,
                  label=f'Required Power = {required_power:.2f}')
        ax.set_xlabel('Simulation number', fontsize=12)
        ax.set_ylabel('Cumulative average power', fontsize=12)
        ax.set_title(f'Power Convergence (effect={bar_effect:.4f})', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Power curves - only if power_curve_effects
    if power_curve_effects and all_power_curves:
        ax = axes[plot_idx]
        
        for i, power_curve_data in enumerate(all_power_curves):
            ax.plot(
                power_curve_effects,
                power_curve_data,
                label=labels[i],
                color=palette[i],
                linewidth=3,
                marker='o',
                markersize=8,
                alpha=0.85
            )
        
        ax.axhline(y=required_power, color='red', linestyle='--', linewidth=2,
                  label=f'Required Power = {required_power:.2f}')
        ax.set_xlabel('Effect size', fontsize=12)
        ax.set_ylabel('Power', fontsize=12)
        ax.set_title('Power Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_multi_test_fwer(
    results: dict,
    alpha: float,
    n_designers: int,
    figsize=None
):
    """
    Visualization of FWER for multiple testing.
    
    Parameters
    ----------
    results : dict
        Results for each correction method.
    alpha : float
        Significance level.
    n_designers : int
        Number of designers (tests).
    figsize : tuple, optional
        Figure size (width, height). Default is (20, 6).
    """
    if figsize is None:
        figsize = (20, 6)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: FWER bars
    ax = axes[0]
    methods_list = list(results.keys())
    y_pos = np.arange(len(methods_list))
    
    for i, method in enumerate(methods_list):
        res = results[method]
        # color = '#2ca02c' if res['fwer'] <= alpha else '#d62728'
        
        ax.barh(
            y_pos[i],
            res['fwer'],
            alpha=0.7,
            #color=color,
            label=method
        )
        ax.errorbar(
            res['fwer'], y_pos[i],
            xerr=[[res['fwer'] - res['fwer_ci_left']],
                  [res['fwer_ci_right'] - res['fwer']]],
            fmt='o', color='black', capsize=5
        )
        ax.text(res['fwer'], y_pos[i], f" {res['fwer']:.4f}", va='center', fontsize=10)
    
    ax.axvline(x=alpha, color='red', linestyle='--', linewidth=2,
              label=f'α = {alpha}')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods_list)
    ax.set_xlabel('FWER (Family-Wise Error Rate)', fontsize=12)
    ax.set_title(f'FWER Comparison ({n_designers} tests)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Plot 2: Convergence
    ax = axes[1]
    for method in methods_list:
        res = results[method]
        violations = np.array(res['violations'])
        cumulative_fwer = np.cumsum(violations) / np.arange(1, len(violations) + 1)
        #color = '#2ca02c' if res['fwer'] <= alpha else '#d62728'
        n_sims = len(res['violations'])
        
        ax.plot(
            np.arange(1, n_sims + 1),
            cumulative_fwer,
            label=f"{method} (FWER={res['fwer']:.3f})",
            linewidth=2,
            alpha=0.85,
            #color=color
        )
    
    ax.axhline(y=alpha, color='red', linestyle='--', linewidth=2,
              label=f'α = {alpha}')
    ax.set_xlabel('Simulation number', fontsize=12)
    ax.set_ylabel('Cumulative FWER', fontsize=12)
    ax.set_title('FWER Convergence', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_multi_test_power(
    results: dict,
    n_designers: int,
    power_curve_effects=None,
    beta: float = 0.20,
    figsize=None
):
    """Visualize power metrics for multiple testing corrections.

    Draws a row of three horizontal-bar charts (Average Power, Any-Pair Power,
    All-Pairs Power) comparing correction methods at the fixed effect sizes
    supplied to :meth:`~BenchMarker.multi_test_power`.

    When ``power_curve_effects`` is provided a second row of three line plots
    shows how each metric evolves as the effect size increases.

    Parameters
    ----------
    results : dict
        Output of :meth:`~BenchMarker.multi_test_power`.  Each value must
        contain ``'avg_power'``, ``'any_pair_power'``, ``'all_pairs_power'``
        scalars, and optionally a ``'power_curve'`` sub-dict with lists of
        the same metrics (one entry per point in ``power_curve_effects``).
    n_designers : int
        Number of designers / simultaneous tests.
    power_curve_effects : array-like, optional
        X-axis values for the power-curve plots.  When ``None`` only the bar
        charts are drawn.
    beta : float, default 0.20
        Target Type-II error rate; ``1 - beta`` is drawn as a reference line
        on power-curve plots.
    figsize : tuple, optional
        Figure size ``(width, height)``.  Defaults to ``(30, 6)`` for bar-only
        output and ``(30, 12)`` when power curves are included.
    """
    has_curves = (
        power_curve_effects is not None
        and any('power_curve' in v for v in results.values())
    )
    n_rows = 2 if has_curves else 1
    if figsize is None:
        figsize = (30, 6 * n_rows)

    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # shape (1, 3) for uniform indexing

    methods_list = list(results.keys())
    y_pos = np.arange(len(methods_list))
    required_power = 1.0 - beta

    # ── Row 0: bar charts ────────────────────────────────────────────────────
    _bar_specs = [
        ('avg_power',       f'Average Power (mean: rejections / {n_designers})', 'Average Power'),
        ('any_pair_power',  'Any-Pair Power (≥1 rejection)',                      'Any-Pair Power'),
        ('all_pairs_power', f'All-Pairs Power (all {n_designers} rejections)',    'All-Pairs Power'),
    ]
    for col, (key, title, xlabel) in enumerate(_bar_specs):
        ax = axes[0, col]
        for i, method in enumerate(methods_list):
            val = results[method][key]
            ax.barh(y_pos[i], val, alpha=0.7, label=method)
            ax.text(val, y_pos[i], f'  {val:.4f}', va='center', fontsize=10)
        ax.axvline(x=required_power, color='red', linestyle='--', linewidth=1.5,
                   label=f'1-β = {required_power:.2f}')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods_list)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)

    # ── Row 1: power curves ──────────────────────────────────────────────────
    if has_curves:
        x = list(power_curve_effects)
        _curve_specs = [
            ('avg_power',       'Average Power',  'Avg Power'),
            ('any_pair_power',  'Any-Pair Power', 'Any-Pair Power'),
            ('all_pairs_power', 'All-Pairs Power','All-Pairs Power'),
        ]
        for col, (key, title, ylabel) in enumerate(_curve_specs):
            ax = axes[1, col]
            for method in methods_list:
                pc = results[method].get('power_curve')
                if pc is None:
                    continue
                ax.plot(x, pc[key], marker='o', markersize=4, linewidth=2,
                        alpha=0.85, label=method)
            ax.axhline(y=required_power, color='red', linestyle='--',
                       linewidth=1.5, label=f'1-β = {required_power:.2f}')
            ax.set_xlabel('Effect size', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'{title} vs Effect Size', fontsize=14, fontweight='bold')
            ax.legend(fontsize=9, loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.show()


def plot_directional_analysis(
    delta1: np.ndarray,
    delta2: np.ndarray,
    disagreement_indices: List[int],
    correlation: float,
    correlation_pvalue: float,
    r_squared: float,
    sign_agreement: float,
    sign_disagreement: float,
    rmse: float,
    mad: float,
    suptitle: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None
):
    """
    Directional analysis visualization comparing two designers.
    
    Creates a 2x2 grid of plots:
    1. Scatter plot (delta2 vs delta1) with sign disagreements highlighted
    2. Residuals histogram (delta2 - delta1)
    3. Bland-Altman plot (mean vs difference)
    4. Statistics summary box
    
    Parameters
    ----------
    delta1 : ndarray
        Effect estimates from designer 1.
    delta2 : ndarray
        Effect estimates from designer 2.
    disagreement_indices : list of int
        Indices where sign(delta1) != sign(delta2).
    correlation : float
        Pearson correlation between delta1 and delta2.
    r_squared : float
        R² (proportion of explained variance).
    sign_agreement : float
        Proportion of simulations with same sign.
    sign_disagreement : float
        Proportion of simulations with different signs.
    rmse : float
        Root mean squared error between methods.
    mad : float
        Mean absolute difference between methods.
    suptitle : str, optional
        Overall title for the figure.
    figsize : tuple, optional
        Figure size (width, height). Default is (20, 16).
    
    Notes
    -----
    The scatter plot uses color coding:
    - Blue points: both methods agree on sign
    - Red crosses: methods disagree on sign (highlighted)
    - Light red background in disagreement quadrants (Q2 and Q4)
    
    R² interpretation:
    - R² ≥ 0.90: Excellent agreement (≥90% variance explained)
    - R² ≥ 0.70: Good agreement (≥70% variance explained)
    - R² < 0.70: Poor agreement
    """
    if figsize is None:
        figsize = (20, 16)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold')
    
    # Prepare data
    same_sign = np.sign(delta1) == np.sign(delta2)
    diff_sign = ~same_sign
    residuals = delta2 - delta1
    mean_vals = (delta1 + delta2) / 2
    diff_vals = delta2 - delta1
    
    # ========== Plot 1: Scatter Plot (delta2 vs delta1) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Determine plot limits
    min_val = min(delta1.min(), delta2.min())
    max_val = max(delta1.max(), delta2.max())
    margin = (max_val - min_val) * 0.1
    plot_min = min_val - margin
    plot_max = max_val + margin
    
    # Highlight disagreement quadrants with more vibrant red background
    # Q2: delta1 < 0, delta2 > 0
    ax1.fill_betweenx([0, plot_max], plot_min, 0,
                       alpha=0.25, color='#ff9999', zorder=0)
    # Q4: delta1 > 0, delta2 < 0
    ax1.fill_betweenx([plot_min, 0], 0, plot_max,
                       alpha=0.25, color='#ff9999', zorder=0)
    
    # Scatter points
    ax1.scatter(delta1[same_sign], delta2[same_sign],
                c='#1f77b4', alpha=0.6, s=30, label='Same sign', zorder=2)
    ax1.scatter(delta1[diff_sign], delta2[diff_sign],
                c='#d62728', alpha=0.9, s=80, marker='x',
                linewidths=2.5, label='Different sign ⚠️', zorder=3)
    
    # Diagonal y=x (perfect agreement)
    ax1.plot([plot_min, plot_max], [plot_min, plot_max],
             'k--', alpha=0.6, linewidth=2, label='y=x (perfect match)', zorder=1)
    
    # Confidence band ±2σ around diagonal
    std_residuals = np.std(residuals)
    x_band = np.array([plot_min, plot_max])
    y_lower = x_band - 2*std_residuals
    y_upper = x_band + 2*std_residuals
    ax1.fill_between(x_band, y_lower, y_upper,
                      alpha=0.15, color='gray', label='±2σ band', zorder=1)
    
    # Quadrant lines
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.4, linewidth=1, zorder=1)
    ax1.axvline(0, color='gray', linestyle=':', alpha=0.4, linewidth=1, zorder=1)
    
    # Quadrant labels
    ax1.text(0.95, 0.95, 'Q1 (++)', transform=ax1.transAxes,
             ha='right', va='top', fontsize=10, alpha=0.6)
    ax1.text(0.05, 0.95, 'Q2 (-+) ⚠️', transform=ax1.transAxes,
             ha='left', va='top', fontsize=10, color='#d62728', fontweight='bold')
    ax1.text(0.05, 0.05, 'Q3 (--)', transform=ax1.transAxes,
             ha='left', va='bottom', fontsize=10, alpha=0.6)
    ax1.text(0.95, 0.05, 'Q4 (+-) ⚠️', transform=ax1.transAxes,
             ha='right', va='bottom', fontsize=10, color='#d62728', fontweight='bold')
    
    ax1.set_xlabel('Designer 1: Effect Estimate (Δ)', fontsize=12)
    ax1.set_ylabel('Designer 2: Effect Estimate (Δ)', fontsize=12)
    ax1.set_title('Agreement Between Methods', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(plot_min, plot_max)
    ax1.set_ylim(plot_min, plot_max)
    
    # ========== Plot 2: Residuals Histogram ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Ensure residuals is a 1D array
    residuals_flat = np.asarray(residuals).flatten()
    ax2.hist(residuals_flat, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2,
                label='Zero (perfect match)', zorder=3)
    ax2.axvline(np.mean(residuals), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(residuals):.4f}', zorder=3)
    
    # Normal distribution overlay
    from scipy.stats import norm
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    hist_height = len(residuals) * (residuals.max() - residuals.min()) / 30
    ax2.plot(x_range,
             norm.pdf(x_range, np.mean(residuals), np.std(residuals)) * hist_height,
             'r-', linewidth=2, label='Normal fit', alpha=0.7)
    
    ax2.set_xlabel('Residuals (Designer 2 - Designer 1)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ========== Plot 3: Bland-Altman Plot ==========
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax3.scatter(mean_vals, diff_vals, alpha=0.6, s=30, color='steelblue')
    
    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)
    
    ax3.axhline(mean_diff, color='blue', linestyle='--', linewidth=2,
                label=f'Mean diff: {mean_diff:.4f}', zorder=3)
    ax3.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', linewidth=1.5,
                label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.4f}', zorder=3)
    ax3.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--', linewidth=1.5,
                label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.4f}', zorder=3)
    ax3.axhline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1, zorder=1)
    
    ax3.set_xlabel('Mean of Two Methods: (Δ₁ + Δ₂) / 2', fontsize=12)
    ax3.set_ylabel('Difference: Δ₂ - Δ₁', fontsize=12)
    ax3.set_title('Bland-Altman Plot', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ========== Plot 4: Statistics Box ==========
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Determine quality indicators (remove emoji to avoid font warnings)
    def get_indicator(value, metric_type):
        """Get colored indicator based on metric value."""
        if metric_type == 'sign_agreement':
            if value >= 0.95:
                return '[EXCELLENT]'
            elif value >= 0.85:
                return '[GOOD]'
            else:
                return '[POOR]'
        elif metric_type == 'correlation':
            if abs(value) >= 0.95:
                return '[EXCELLENT]'
            elif abs(value) >= 0.80:
                return '[GOOD]'
            else:
                return '[POOR]'
        elif metric_type == 'sign_disagreement':
            if value <= 0.05:
                return '[EXCELLENT]'
            elif value <= 0.15:
                return '[ACCEPTABLE]'
            else:
                return '[HIGH]'
        return ''
    
    # Overall interpretation
    if sign_agreement >= 0.95 and abs(correlation) >= 0.95:
        interpretation = "[EXCELLENT] Methods show nearly identical results"
    elif sign_agreement >= 0.85 and abs(correlation) >= 0.80:
        interpretation = "[GOOD] Methods are well aligned"
    elif sign_disagreement > 0.15:
        interpretation = "[CRITICAL] High rate of sign disagreements!"
    elif abs(correlation) < 0.50:
        interpretation = "[POOR] Weak relationship between methods"
    else:
        interpretation = "[FAIR] Methods partially aligned"
    
    # Format statistics text with proper alignment
    stats_lines = [
        "=" * 60,
        "DIRECTIONAL ANALYSIS STATISTICS".center(60),
        "=" * 60,
        "",
        "Sign Agreement:",
        f"  Agreement Rate:      {sign_agreement:.3f}  {get_indicator(sign_agreement, 'sign_agreement')}",
        f"  Disagreement Rate:   {sign_disagreement:.3f}  {get_indicator(sign_disagreement, 'sign_disagreement')}",
        f"  Disagreements:       {len(disagreement_indices)} / {len(delta1)}",
        "",
        "Pearson Correlation:",
        f"  Correlation (r):     {correlation:.3f}  {get_indicator(correlation, 'correlation')}",
        f"  R-squared (r^2):     {r_squared:.3f}",
        f"  P-value:             {correlation_pvalue:.4e}",
        f"  Significant:         {'YES' if correlation_pvalue < 0.05 else 'NO'} (alpha=0.05)",
        "",
        "Dispersion Metrics:",
        f"  RMSE:                {rmse:.4f}",
        f"  MAD:                 {mad:.4f}",
        f"  Residuals SD:        {np.std(residuals):.4f}",
        "",
        "Interpretation:",
        f"  {interpretation}",
        "",
        "=" * 60,
        "",
        "Correlation Interpretation:",
        f"  r = {correlation:.3f} indicates",
        f"  {'strong' if abs(correlation) >= 0.8 else 'moderate' if abs(correlation) >= 0.5 else 'weak'} {'positive' if correlation > 0 else 'negative'} relationship",
        "  |r| >= 0.95: Excellent agreement",
        "  |r| >= 0.80: Good agreement",
        "  |r| <  0.80: Poor agreement",
    ]
    
    stats_text = "\n".join(stats_lines)
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=10, family='monospace', va='top', ha='left',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9, pad=1.5))
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()
