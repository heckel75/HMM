"""
============================================================================
DATE-BASED LOTTERY PATTERN DETECTION
============================================================================

These methods use the draw DATE to detect temporal rigging patterns
that pure number analysis misses.

METHODS:
  D1. Day-of-Week Bias
  D2. Monthly / Seasonal Patterns
  D3. Draw Spacing Regularity
  D4. Date-Number Correlation (do numbers correlate with calendar?)
  D5. Trend Analysis (drift over time)
  D6. Day-of-Month Influence
  D7. Year-over-Year Comparison
  D8. Before/After Event Split (split at arbitrary dates)
  D9. Weekday vs Weekend Comparison
  D10. Lunar Phase Analysis (surprisingly common in rigging folklore)
  D11. Number Sum vs Calendar Features (multivariate)
  D12. Rolling Window Anomaly Detection

============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chisquare, ks_2samp, pearsonr, spearmanr
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# DATA LOADING
# =====================================================================

def load_data_with_dates(filepath='Data.csv', sep=';',
                          date_col='date', number_cols=None,
                          date_format='%d/%m/%Y'):
    """Load lottery data with date parsing."""
    df = pd.read_csv(filepath, sep=sep)
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    df = df.sort_values(date_col).reset_index(drop=True)
    
    if number_cols is None:
        number_cols = [c for c in df.columns if c != date_col]
    
    draws = df[number_cols].values
    dates = df[date_col]
    
    print(f"  Loaded {len(draws)} draws")
    print(f"  Date range: {dates.iloc[0].strftime('%d/%m/%Y')} → {dates.iloc[-1].strftime('%d/%m/%Y')}")
    print(f"  Span: {(dates.iloc[-1] - dates.iloc[0]).days} days")
    print(f"  Numbers: {draws.shape[1]} per draw, range {draws.min()}-{draws.max()}")
    
    return draws, dates, df


# =====================================================================
# D1: DAY OF WEEK BIAS
# =====================================================================

def method_d01_day_of_week(draws, dates, max_number=49):
    """
    WHAT IT TESTS:
      Do draws on different weekdays have different statistical properties?

    HOW IT WORKS:
      Group draws by weekday. Compare number distributions, means,
      and frequencies across days.

    WHAT RIGGING IT CATCHES:
      Day-dependent manipulation (e.g., rigging only on low-audience days).
    """
    print("\n" + "=" * 70)
    print("METHOD D1: DAY-OF-WEEK ANALYSIS")
    print("=" * 70)

    days = dates.dt.day_name()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    print(f"\n  Draw frequency by day:")
    day_counts = days.value_counts()
    for day in day_names:
        count = day_counts.get(day, 0)
        if count > 0:
            print(f"    {day:>10s}: {count:4d} draws")

    # Compare draw sum distributions across active days
    active_days = [d for d in day_names if day_counts.get(d, 0) > 10]
    day_sums = {}
    for day in active_days:
        mask = days == day
        day_sums[day] = draws[mask].sum(axis=1)

    if len(active_days) >= 2:
        # Kruskal-Wallis test (non-parametric ANOVA)
        groups = [day_sums[d] for d in active_days]
        h_stat, h_p = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis test (draw sums across days):")
        print(f"  H = {h_stat:.4f}, p = {h_p:.6f}")
        print(f"  {'⚠️  DAY-DEPENDENT PATTERNS' if h_p < 0.05 else '✓ No day-of-week effect'}")

        # Per-day statistics
        print(f"\n  Per-day draw sum statistics:")
        for day in active_days:
            s = day_sums[day]
            print(f"    {day:>10s}: mean={np.mean(s):.1f}, std={np.std(s):.1f}, "
                  f"median={np.median(s):.1f}")

        # Per-day number frequency comparison
        print(f"\n  Per-day most common numbers:")
        for day in active_days:
            mask = days == day
            day_draws = draws[mask].flatten()
            freq = Counter(day_draws)
            top5 = freq.most_common(5)
            nums = [f"{n}({c})" for n, c in top5]
            print(f"    {day:>10s}: {', '.join(nums)}")

        # Pairwise comparison between days
        print(f"\n  Pairwise KS tests between days:")
        suspicious_pairs = 0
        for i, d1 in enumerate(active_days):
            for d2 in active_days[i+1:]:
                ks_stat, ks_p = ks_2samp(day_sums[d1], day_sums[d2])
                if ks_p < 0.05:
                    suspicious_pairs += 1
                    print(f"    ⚠️  {d1} vs {d2}: KS={ks_stat:.4f}, p={ks_p:.4f}")

        if suspicious_pairs == 0:
            print(f"    No significant differences between any day pairs")

        return {'h_stat': h_stat, 'h_p': h_p, 'suspicious_pairs': suspicious_pairs}

    return {}


# =====================================================================
# D2: MONTHLY / SEASONAL PATTERNS
# =====================================================================

def method_d02_monthly(draws, dates, max_number=49):
    """
    WHAT IT TESTS:
      Do draws differ by month or season?

    WHAT RIGGING IT CATCHES:
      Seasonal rigging, quarterly manipulation cycles.
    """
    print("\n" + "=" * 70)
    print("METHOD D2: MONTHLY / SEASONAL ANALYSIS")
    print("=" * 70)

    months = dates.dt.month
    draw_sums = draws.sum(axis=1)

    # Monthly statistics
    print(f"\n  Monthly draw statistics:")
    monthly_sums = {}
    for m in range(1, 13):
        mask = months == m
        if mask.sum() > 0:
            s = draw_sums[mask]
            monthly_sums[m] = s
            month_name = dates[mask].dt.month_name().iloc[0]
            print(f"    {month_name:>10s} (n={mask.sum():3d}): "
                  f"mean={np.mean(s):.1f}, std={np.std(s):.1f}")

    # Kruskal-Wallis across months
    active_months = [m for m in range(1, 13) if (months == m).sum() > 5]
    if len(active_months) >= 2:
        groups = [draw_sums[months == m] for m in active_months]
        h_stat, h_p = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis across months: H={h_stat:.4f}, p={h_p:.6f}")
        print(f"  {'⚠️  MONTHLY VARIATION DETECTED' if h_p < 0.05 else '✓ No monthly effect'}")

    # Seasonal (quarterly) comparison
    quarters = dates.dt.quarter
    print(f"\n  Quarterly comparison:")
    for q in range(1, 5):
        mask = quarters == q
        if mask.sum() > 0:
            s = draw_sums[mask]
            print(f"    Q{q} (n={mask.sum():3d}): mean={np.mean(s):.1f}, std={np.std(s):.1f}")

    q_groups = [draw_sums[quarters == q] for q in range(1, 5) if (quarters == q).sum() > 5]
    if len(q_groups) >= 2:
        h_q, p_q = stats.kruskal(*q_groups)
        print(f"  Kruskal-Wallis across quarters: H={h_q:.4f}, p={p_q:.6f}")
        print(f"  {'⚠️  SEASONAL PATTERN' if p_q < 0.05 else '✓ No seasonal effect'}")

    return {'h_p': h_p if len(active_months) >= 2 else 1.0}


# =====================================================================
# D3: DRAW SPACING ANALYSIS
# =====================================================================

def method_d03_spacing(draws, dates):
    """
    WHAT IT TESTS:
      Are draws evenly spaced? Do irregular gaps correlate with anomalies?

    WHAT RIGGING IT CATCHES:
      Extra draws inserted, draws skipped to manipulate outcomes,
      schedule manipulation.
    """
    print("\n" + "=" * 70)
    print("METHOD D3: DRAW SPACING ANALYSIS")
    print("=" * 70)

    gaps = dates.diff().dt.days.dropna()
    draw_sums = draws.sum(axis=1)

    print(f"\n  Gap statistics (days between draws):")
    print(f"    Mean: {gaps.mean():.2f} days")
    print(f"    Median: {gaps.median():.1f} days")
    print(f"    Std: {gaps.std():.2f} days")
    print(f"    Min: {gaps.min():.0f} days")
    print(f"    Max: {gaps.max():.0f} days")

    # Distribution of gaps
    gap_counts = Counter(gaps.astype(int))
    print(f"\n  Gap distribution:")
    for gap_val in sorted(gap_counts.keys()):
        count = gap_counts[gap_val]
        print(f"    {gap_val:3.0f} days: {count:4d} times ({count/len(gaps)*100:.1f}%)")

    # Test: do draws after unusual gaps have different properties?
    median_gap = gaps.median()
    long_gap_mask = np.array([False] + list(gaps > median_gap * 1.5))
    short_gap_mask = np.array([False] + list(gaps < median_gap * 0.5))

    if long_gap_mask.sum() > 5 and (~long_gap_mask).sum() > 5:
        long_sums = draw_sums[long_gap_mask[:len(draw_sums)]]
        normal_sums = draw_sums[~long_gap_mask[:len(draw_sums)]]
        ks_stat, ks_p = ks_2samp(long_sums, normal_sums)
        print(f"\n  Draws after long gaps vs normal gaps:")
        print(f"    Long gap draws (n={len(long_sums)}): mean sum={np.mean(long_sums):.1f}")
        print(f"    Normal draws (n={len(normal_sums)}): mean sum={np.mean(normal_sums):.1f}")
        print(f"    KS test: p={ks_p:.6f}")
        print(f"    {'⚠️  DRAWS DIFFER AFTER LONG GAPS' if ks_p < 0.05 else '✓ No gap-dependent effect'}")

    # Correlation between gap length and next draw sum
    if len(gaps) > 10:
        corr, p_corr = pearsonr(gaps.values, draw_sums[1:len(gaps)+1])
        print(f"\n  Correlation (gap length vs next draw sum):")
        print(f"    Pearson r = {corr:.4f}, p = {p_corr:.6f}")
        print(f"    {'⚠️  GAP INFLUENCES DRAWS' if p_corr < 0.05 else '✓ No correlation'}")

    return {}


# =====================================================================
# D4: DATE-NUMBER CORRELATION
# =====================================================================

def method_d04_date_number(draws, dates, max_number=49):
    """
    WHAT IT TESTS:
      Do drawn numbers correlate with calendar features?
      (e.g., does the day of month appear more often as a drawn number?)

    WHAT RIGGING IT CATCHES:
      Human-influenced "meaningful" number selection,
      date-seeded pseudo-random generators.
    """
    print("\n" + "=" * 70)
    print("METHOD D4: DATE-NUMBER CORRELATION")
    print("=" * 70)

    day_of_month = dates.dt.day.values
    week_of_year = dates.dt.isocalendar().week.values.astype(int)
    month = dates.dt.month.values
    year = dates.dt.year.values

    # Test 1: Does the day-of-month appear in the draw more than expected?
    dom_in_draw = 0
    dom_possible = 0
    for i, draw in enumerate(draws):
        dom = day_of_month[i]
        if dom <= max_number:
            dom_possible += 1
            if dom in draw:
                dom_in_draw += 1

    expected_rate = draws.shape[1] / max_number
    observed_rate = dom_in_draw / dom_possible if dom_possible > 0 else 0
    p_binom = stats.binomtest(dom_in_draw, dom_possible, expected_rate).pvalue

    print(f"\n  Test 1: Day-of-month appearing in draw")
    print(f"    Observed: {dom_in_draw}/{dom_possible} ({observed_rate:.1%})")
    print(f"    Expected: {expected_rate:.1%}")
    print(f"    p-value: {p_binom:.6f}")
    print(f"    {'⚠️  DAY-OF-MONTH BIAS' if p_binom < 0.05 else '✓ No day-of-month bias'}")

    # Test 2: Does the month number appear more?
    month_in_draw = sum(1 for i, draw in enumerate(draws) if month[i] in draw)
    expected_month = len(draws) * draws.shape[1] / max_number
    p_month = stats.binomtest(month_in_draw, len(draws), draws.shape[1] / max_number).pvalue

    print(f"\n  Test 2: Month number appearing in draw")
    print(f"    Observed: {month_in_draw}/{len(draws)} ({month_in_draw/len(draws):.1%})")
    print(f"    Expected: {expected_rate:.1%}")
    print(f"    p-value: {p_month:.6f}")
    print(f"    {'⚠️  MONTH-NUMBER BIAS' if p_month < 0.05 else '✓ No month-number bias'}")

    # Test 3: Correlation between draw sum and day/week/month
    draw_sums = draws.sum(axis=1)
    print(f"\n  Test 3: Correlations between calendar and draw sums")
    for name, values in [("Day of month", day_of_month),
                          ("Week of year", week_of_year),
                          ("Month", month)]:
        corr, p = spearmanr(values, draw_sums)
        flag = "⚠️" if p < 0.05 else "  "
        print(f"    {flag} {name:>15s}: Spearman ρ={corr:+.4f}, p={p:.6f}")

    # Test 4: Does the year influence draws?
    if len(set(year)) > 1:
        print(f"\n  Test 4: Year-by-year draw sum comparison")
        for y in sorted(set(year)):
            mask = year == y
            s = draw_sums[mask]
            print(f"    {y}: n={mask.sum():3d}, mean sum={np.mean(s):.1f}, std={np.std(s):.1f}")

    # Test 5: Sum of date digits correlation
    date_digit_sums = []
    for d in dates:
        digits = d.strftime('%d%m%Y')
        date_digit_sums.append(sum(int(c) for c in digits))
    corr_dds, p_dds = spearmanr(date_digit_sums, draw_sums)
    print(f"\n  Test 5: Date digit sum vs draw sum")
    print(f"    Spearman ρ={corr_dds:+.4f}, p={p_dds:.6f}")
    print(f"    {'⚠️  DATE-SEEDED PATTERN' if p_dds < 0.05 else '✓ No date-seed effect'}")

    return {'dom_p': p_binom, 'month_p': p_month}


# =====================================================================
# D5: TREND ANALYSIS
# =====================================================================

def method_d05_trend(draws, dates):
    """
    WHAT IT TESTS:
      Is there a systematic drift in draw properties over time?

    WHAT RIGGING IT CATCHES:
      Gradual shift in manipulation, changing rigging strategy,
      machine degradation/calibration drift.
    """
    print("\n" + "=" * 70)
    print("METHOD D5: TREND ANALYSIS (Drift Over Time)")
    print("=" * 70)

    draw_sums = draws.sum(axis=1)
    draw_index = np.arange(len(draws))

    # Linear regression on draw sums over time
    slope, intercept, r_value, p_value, std_err = stats.linregress(draw_index, draw_sums)

    print(f"\n  Linear trend in draw sums:")
    print(f"    Slope: {slope:.4f} per draw")
    print(f"    R²: {r_value**2:.6f}")
    print(f"    p-value: {p_value:.6f}")
    print(f"    {'⚠️  SIGNIFICANT TREND' if p_value < 0.05 else '✓ No trend'}")

    # Test trend in individual number frequencies (rolling window)
    print(f"\n  Rolling frequency analysis (window=100 draws):")
    window = 100
    max_number = draws.max()
    trending_numbers = []

    for num in range(1, max_number + 1):
        presence = np.array([1.0 if num in draw else 0.0 for draw in draws])
        rolling_freq = np.convolve(presence, np.ones(window)/window, mode='valid')

        if len(rolling_freq) > 10:
            idx = np.arange(len(rolling_freq))
            s, _, r, p, _ = stats.linregress(idx, rolling_freq)
            if p < 0.01:
                direction = "↑ increasing" if s > 0 else "↓ decreasing"
                trending_numbers.append((num, s, p, direction))
                print(f"    ⚠️  Number {num:2d}: {direction}, slope={s:.6f}, p={p:.4f}")

    if not trending_numbers:
        print(f"    No numbers with significant trends")

    # Mann-Kendall trend test on draw sums
    print(f"\n  Mann-Kendall trend test on draw sums:")
    n = len(draw_sums)
    s_mk = 0
    for i in range(n - 1):
        for j in range(i + 1, min(i + 50, n)):  # Limit for performance
            s_mk += np.sign(draw_sums[j] - draw_sums[i])

    print(f"    S statistic: {s_mk}")
    print(f"    {'⚠️  TREND DETECTED' if abs(s_mk) > 2 * n else '✓ No monotonic trend'}")

    return {'trend_p': p_value, 'trending_numbers': trending_numbers}


# =====================================================================
# D6: DAY-OF-MONTH INFLUENCE
# =====================================================================

def method_d06_day_of_month(draws, dates, max_number=49):
    """
    WHAT IT TESTS:
      Do draws on different days of the month behave differently?
      (e.g., 1st of month vs 15th vs 28th)

    WHAT RIGGING IT CATCHES:
      Calendar-based rigging schedules, pay-day effects.
    """
    print("\n" + "=" * 70)
    print("METHOD D6: DAY-OF-MONTH INFLUENCE")
    print("=" * 70)

    dom = dates.dt.day
    draw_sums = draws.sum(axis=1)

    # Group by day-of-month
    dom_groups = {}
    for d in range(1, 32):
        mask = dom == d
        if mask.sum() >= 3:
            dom_groups[d] = draw_sums[mask]

    # Kruskal-Wallis across days
    if len(dom_groups) >= 2:
        groups = list(dom_groups.values())
        h_stat, h_p = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis across days of month: H={h_stat:.4f}, p={h_p:.6f}")
        print(f"  {'⚠️  DAY-OF-MONTH EFFECT' if h_p < 0.05 else '✓ No day-of-month effect'}")

    # Early vs mid vs late month
    early = draw_sums[dom <= 10]
    mid = draw_sums[(dom > 10) & (dom <= 20)]
    late = draw_sums[dom > 20]

    print(f"\n  Early month (1-10):  n={len(early):3d}, mean={np.mean(early):.1f}")
    print(f"  Mid month (11-20):   n={len(mid):3d}, mean={np.mean(mid):.1f}")
    print(f"  Late month (21-31):  n={len(late):3d}, mean={np.mean(late):.1f}")

    h2, p2 = stats.kruskal(early, mid, late)
    print(f"  Kruskal-Wallis (early/mid/late): H={h2:.4f}, p={p2:.6f}")

    return {'h_p': h_p if len(dom_groups) >= 2 else 1.0}


# =====================================================================
# D7: YEAR-OVER-YEAR COMPARISON
# =====================================================================

def method_d07_yearly(draws, dates, max_number=49):
    """
    WHAT IT TESTS:
      Did the lottery's statistical properties change across years?

    WHAT RIGGING IT CATCHES:
      Rigging starting/stopping at a certain point,
      rule changes, equipment changes.
    """
    print("\n" + "=" * 70)
    print("METHOD D7: YEAR-OVER-YEAR COMPARISON")
    print("=" * 70)

    years = dates.dt.year
    draw_sums = draws.sum(axis=1)

    yearly_data = {}
    for y in sorted(years.unique()):
        mask = years == y
        if mask.sum() >= 10:
            yearly_data[y] = {
                'sums': draw_sums[mask],
                'draws': draws[mask],
                'n': mask.sum()
            }

    print(f"\n  Year-by-year statistics:")
    for y, data in yearly_data.items():
        s = data['sums']
        # Number frequency chi-squared within year
        flat = data['draws'].flatten()
        freq = np.bincount(flat, minlength=max_number + 1)[1:]
        expected = np.full(max_number, len(flat) / max_number)
        chi2, p = chisquare(freq, expected)
        print(f"    {y}: n={data['n']:3d}, mean_sum={np.mean(s):.1f}, "
              f"std={np.std(s):.1f}, freq_χ²_p={p:.4f}")

    # Pairwise KS test between consecutive years
    year_list = sorted(yearly_data.keys())
    if len(year_list) >= 2:
        print(f"\n  Consecutive year comparisons:")
        for i in range(len(year_list) - 1):
            y1, y2 = year_list[i], year_list[i + 1]
            ks, p = ks_2samp(yearly_data[y1]['sums'], yearly_data[y2]['sums'])
            flag = "⚠️" if p < 0.05 else "  "
            print(f"    {flag} {y1} vs {y2}: KS={ks:.4f}, p={p:.4f}")

    # Overall Kruskal-Wallis across years
    if len(yearly_data) >= 2:
        groups = [d['sums'] for d in yearly_data.values()]
        h, p = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis across all years: H={h:.4f}, p={p:.6f}")
        print(f"  {'⚠️  YEARLY VARIATION DETECTED' if p < 0.05 else '✓ Consistent across years'}")
        return {'h_p': p}

    return {}


# =====================================================================
# D8: STRUCTURAL BREAK DETECTION
# =====================================================================

def method_d08_structural_break(draws, dates):
    """
    WHAT IT TESTS:
      Is there a point in time where the lottery's behavior changed?

    HOW IT WORKS:
      Test every possible split point using the Chow test approach.
      Find the date that maximizes the statistical difference.

    WHAT RIGGING IT CATCHES:
      Start/stop of rigging, equipment change, personnel change.
    """
    print("\n" + "=" * 70)
    print("METHOD D8: STRUCTURAL BREAK DETECTION")
    print("=" * 70)

    draw_sums = draws.sum(axis=1)
    n = len(draw_sums)
    min_segment = max(30, n // 10)

    best_stat = 0
    best_idx = 0
    all_stats = []

    for split in range(min_segment, n - min_segment):
        before = draw_sums[:split]
        after = draw_sums[split:]
        ks_stat, _ = ks_2samp(before, after)
        all_stats.append(ks_stat)
        if ks_stat > best_stat:
            best_stat = ks_stat
            best_idx = split

    # Significance via permutation
    perm_max_stats = []
    for _ in range(500):
        perm = np.random.permutation(draw_sums)
        max_ks = 0
        for split in range(min_segment, n - min_segment, 5):  # Coarse for speed
            ks_s, _ = ks_2samp(perm[:split], perm[split:])
            max_ks = max(max_ks, ks_s)
        perm_max_stats.append(max_ks)

    p_value = np.mean(np.array(perm_max_stats) >= best_stat)

    best_date = dates.iloc[best_idx].strftime('%d/%m/%Y')
    print(f"\n  Most likely break point: Draw {best_idx} ({best_date})")
    print(f"  KS statistic at break: {best_stat:.4f}")
    print(f"  Permutation p-value: {p_value:.4f}")

    # Statistics before and after
    before = draw_sums[:best_idx]
    after = draw_sums[best_idx:]
    print(f"\n  Before ({best_idx} draws): mean={np.mean(before):.1f}, std={np.std(before):.1f}")
    print(f"  After ({n-best_idx} draws):  mean={np.mean(after):.1f}, std={np.std(after):.1f}")
    print(f"\n  {'⚠️  STRUCTURAL BREAK DETECTED' if p_value < 0.05 else '✓ No structural break'}")

    return {'break_date': best_date, 'break_idx': best_idx, 'p_value': p_value}


# =====================================================================
# D9: WEEKDAY vs WEEKEND
# =====================================================================

def method_d09_weekday_weekend(draws, dates, max_number=49):
    """
    WHAT IT TESTS:
      Do weekday vs weekend draws differ?
    """
    print("\n" + "=" * 70)
    print("METHOD D9: WEEKDAY vs WEEKEND")
    print("=" * 70)

    is_weekend = dates.dt.dayofweek >= 5
    draw_sums = draws.sum(axis=1)

    wd_sums = draw_sums[~is_weekend]
    we_sums = draw_sums[is_weekend]

    print(f"\n  Weekday draws: {len(wd_sums)}, mean sum={np.mean(wd_sums):.1f}")
    print(f"  Weekend draws: {len(we_sums)}, mean sum={np.mean(we_sums):.1f}")

    if len(we_sums) > 5 and len(wd_sums) > 5:
        ks, p = ks_2samp(wd_sums, we_sums)
        mw_stat, mw_p = stats.mannwhitneyu(wd_sums, we_sums, alternative='two-sided')
        print(f"\n  KS test: statistic={ks:.4f}, p={p:.6f}")
        print(f"  Mann-Whitney U test: statistic={mw_stat:.1f}, p={mw_p:.6f}")
        print(f"  {'⚠️  WEEKDAY/WEEKEND DIFFER' if min(p, mw_p) < 0.05 else '✓ No difference'}")

        # Number frequency comparison
        wd_flat = draws[~is_weekend].flatten()
        we_flat = draws[is_weekend].flatten()
        wd_freq = np.bincount(wd_flat, minlength=max_number+1)[1:].astype(float)
        we_freq = np.bincount(we_flat, minlength=max_number+1)[1:].astype(float)
        wd_freq /= wd_freq.sum()
        we_freq /= we_freq.sum()

        max_diff_num = np.argmax(np.abs(wd_freq - we_freq)) + 1
        print(f"\n  Number with biggest weekday/weekend difference: {max_diff_num}")
        print(f"    Weekday freq: {wd_freq[max_diff_num-1]:.4f}")
        print(f"    Weekend freq: {we_freq[max_diff_num-1]:.4f}")
    else:
        print(f"  Not enough weekend draws for comparison")

    return {}


# =====================================================================
# D10: LUNAR PHASE ANALYSIS
# =====================================================================

def method_d10_lunar(draws, dates):
    """
    WHAT IT TESTS:
      Do draws correlate with lunar phases?

    NOTE:
      Included for completeness. Unlikely to reveal rigging but
      demonstrates thoroughness and can catch date-based patterns.
    """
    print("\n" + "=" * 70)
    print("METHOD D10: LUNAR PHASE ANALYSIS")
    print("=" * 70)

    # Simple lunar phase approximation (synodic period ≈ 29.53 days)
    # Reference: Jan 6, 2000 was a new moon
    reference_new_moon = pd.Timestamp('2000-01-06')
    synodic_period = 29.53

    days_since_ref = (dates - reference_new_moon).dt.total_seconds() / 86400
    lunar_phase = (days_since_ref % synodic_period) / synodic_period

    # Categorize: New (0-0.125), Waxing Crescent, First Quarter, etc.
    phase_names = ['New Moon', 'Waxing Crescent', 'First Quarter', 'Waxing Gibbous',
                   'Full Moon', 'Waning Gibbous', 'Last Quarter', 'Waning Crescent']
    phase_idx = (lunar_phase * 8).astype(int) % 8

    draw_sums = draws.sum(axis=1)

    print(f"\n  Draws by lunar phase:")
    phase_sums = {}
    for i, name in enumerate(phase_names):
        mask = phase_idx == i
        if mask.sum() > 0:
            s = draw_sums[mask]
            phase_sums[name] = s
            print(f"    {name:>18s}: n={mask.sum():3d}, mean_sum={np.mean(s):.1f}")

    if len(phase_sums) >= 2:
        groups = list(phase_sums.values())
        h, p = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis across lunar phases: H={h:.4f}, p={p:.6f}")
        print(f"  {'⚠️  LUNAR CORRELATION (!?)' if p < 0.05 else '✓ No lunar effect (as expected)'}")

    return {}


# =====================================================================
# D11: MULTIVARIATE DATE-NUMBER ANALYSIS
# =====================================================================

def method_d11_multivariate(draws, dates, max_number=49):
    """
    WHAT IT TESTS:
      Can calendar features jointly predict draw properties?
      Uses multiple calendar features simultaneously.

    WHAT RIGGING IT CATCHES:
      Complex date-dependent algorithms.
    """
    print("\n" + "=" * 70)
    print("METHOD D11: MULTIVARIATE DATE-NUMBER ANALYSIS")
    print("=" * 70)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    # Build calendar features
    X = pd.DataFrame({
        'day_of_week': dates.dt.dayofweek,
        'day_of_month': dates.dt.day,
        'month': dates.dt.month,
        'week_of_year': dates.dt.isocalendar().week.astype(int),
        'day_of_year': dates.dt.dayofyear,
        'is_weekend': (dates.dt.dayofweek >= 5).astype(int),
        'quarter': dates.dt.quarter,
    }).values

    draw_sums = draws.sum(axis=1)
    y_binary = (draw_sums > np.median(draw_sums)).astype(int)

    # Can calendar predict draw sum?
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    scores = cross_val_score(rf, X, y_binary, cv=10, scoring='accuracy')

    print(f"\n  Can calendar features predict draw sum above/below median?")
    print(f"  Features: day_of_week, day_of_month, month, week, day_of_year, weekend, quarter")
    print(f"  Baseline: 50%")
    print(f"  Random Forest accuracy: {scores.mean():.1%} ± {scores.std():.1%}")
    print(f"  {'⚠️  DATE PREDICTS DRAWS' if scores.mean() > 0.55 else '✓ Calendar has no predictive power'}")

    # Feature importance
    rf.fit(X, y_binary)
    feature_names = ['day_of_week', 'day_of_month', 'month', 'week_of_year',
                     'day_of_year', 'is_weekend', 'quarter']
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n  Feature importances:")
    for idx in sorted_idx:
        print(f"    {feature_names[idx]:>15s}: {importances[idx]:.4f}")

    # Per-number: can calendar predict if specific number appears?
    print(f"\n  Per-number calendar predictability:")
    predictable_numbers = []
    for num in range(1, max_number + 1):
        y_num = np.array([1 if num in draw else 0 for draw in draws])
        baseline = max(y_num.mean(), 1 - y_num.mean())
        try:
            rf_num = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
            s = cross_val_score(rf_num, X, y_num, cv=5, scoring='accuracy')
            improvement = s.mean() - baseline
            if improvement > 0.03:
                predictable_numbers.append((num, s.mean(), baseline))
                print(f"    ⚠️  Number {num:2d}: accuracy={s.mean():.1%}, baseline={baseline:.1%}")
        except:
            continue

    if not predictable_numbers:
        print(f"    No numbers predictable from calendar features")

    return {'accuracy': scores.mean()}


# =====================================================================
# D12: ROLLING WINDOW ANOMALY DETECTION
# =====================================================================

def method_d12_rolling_anomaly(draws, dates, max_number=49):
    """
    WHAT IT TESTS:
      Are there specific time windows where the lottery behaves anomalously?

    HOW IT WORKS:
      Slide a window across time, test each window against the overall
      distribution. Flag windows that deviate significantly.

    WHAT RIGGING IT CATCHES:
      Intermittent rigging, specific event-triggered manipulation.
    """
    print("\n" + "=" * 70)
    print("METHOD D12: ROLLING WINDOW ANOMALY DETECTION")
    print("=" * 70)

    draw_sums = draws.sum(axis=1)
    n = len(draws)
    window_size = 50
    step = 10

    print(f"\n  Window size: {window_size} draws, step: {step}")
    print(f"  Testing {(n - window_size) // step + 1} windows\n")

    anomalous_windows = []

    for start in range(0, n - window_size, step):
        end = start + window_size
        window_sums = draw_sums[start:end]
        rest_sums = np.concatenate([draw_sums[:start], draw_sums[end:]])

        if len(rest_sums) < window_size:
            continue

        ks_stat, ks_p = ks_2samp(window_sums, rest_sums)

        if ks_p < 0.01:  # Strict threshold
            date_start = dates.iloc[start].strftime('%d/%m/%Y')
            date_end = dates.iloc[end-1].strftime('%d/%m/%Y')
            anomalous_windows.append({
                'start': start, 'end': end,
                'date_start': date_start, 'date_end': date_end,
                'ks_stat': ks_stat, 'p': ks_p,
                'mean': np.mean(window_sums),
                'overall_mean': np.mean(draw_sums)
            })

    if anomalous_windows:
        print(f"  ⚠️  Found {len(anomalous_windows)} anomalous time windows:")
        for w in anomalous_windows:
            direction = "HIGH" if w['mean'] > w['overall_mean'] else "LOW"
            print(f"    {w['date_start']} → {w['date_end']}: "
                  f"mean={w['mean']:.1f} ({direction}), p={w['p']:.4f}")
    else:
        print(f"  ✓ No anomalous time windows detected")

    # Also check for anomalous number frequency windows
    print(f"\n  Per-number rolling anomaly check:")
    number_anomalies = []
    for num in range(1, max_number + 1):
        presence = np.array([1.0 if num in draw else 0.0 for draw in draws])
        overall_rate = presence.mean()

        for start in range(0, n - window_size, step):
            window_rate = presence[start:start+window_size].mean()
            # Binomial test
            k = int(presence[start:start+window_size].sum())
            p = stats.binomtest(k, window_size, overall_rate).pvalue
            if p < 0.001:  # Very strict
                date_s = dates.iloc[start].strftime('%d/%m/%Y')
                date_e = dates.iloc[start+window_size-1].strftime('%d/%m/%Y')
                number_anomalies.append((num, date_s, date_e, window_rate, overall_rate, p))

    if number_anomalies:
        print(f"  ⚠️  Found {len(number_anomalies)} number-specific anomalies:")
        for num, ds, de, wr, oar, p in number_anomalies[:15]:
            direction = "↑" if wr > oar else "↓"
            print(f"    Number {num:2d} ({ds}→{de}): rate={wr:.1%} vs overall {oar:.1%} {direction}, p={p:.6f}")
    else:
        print(f"  ✓ No number-specific anomalies in any window")

    return {'n_anomalous_windows': len(anomalous_windows),
            'n_number_anomalies': len(number_anomalies)}


# =====================================================================
# SUMMARY
# =====================================================================

def print_date_summary(results):
    """Print summary of date-based analysis."""
    print("\n\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "  DATE-BASED ANALYSIS SUMMARY  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    print(f"""
┌──────────────────────────────────────────────────────────────────┐
│ METHOD                              │ KEY FINDING               │
├─────────────────────────────────────┼───────────────────────────┤""")

    methods = [
        ("D1.  Day-of-Week",         results.get('d01', {}).get('h_p')),
        ("D2.  Monthly/Seasonal",    results.get('d02', {}).get('h_p')),
        ("D4.  Date-Number Corr",    results.get('d04', {}).get('dom_p')),
        ("D5.  Trend (Drift)",       results.get('d05', {}).get('trend_p')),
        ("D6.  Day-of-Month",        results.get('d06', {}).get('h_p')),
        ("D7.  Year-over-Year",      results.get('d07', {}).get('h_p')),
        ("D8.  Structural Break",    results.get('d08', {}).get('p_value')),
        ("D11. Calendar ML",         results.get('d11', {}).get('accuracy')),
    ]

    suspicious = 0
    for name, value in methods:
        if value is None:
            status = "  --  "
        elif "ML" in name:
            is_sus = value > 0.55
            status = f"{'⚠️' if is_sus else '✓'}  acc={value:.1%}"
        else:
            is_sus = value < 0.05
            status = f"{'⚠️' if is_sus else '✓'}  p={value:.4f}"
        if value is not None and is_sus:
            suspicious += 1
        print(f"│ {name:<37s}│ {status:<25s}  │")

    print(f"└─────────────────────────────────────┴───────────────────────────┘")
    print(f"\n  Date-based anomalies: {suspicious}/{len(methods)}")


# =====================================================================
# MAIN
# =====================================================================

def run_date_analysis(filepath='Data.csv', sep=';',
                       date_col='date', number_cols=None,
                       max_number=49, date_format='%d/%m/%Y'):
    """Run all date-based analysis methods."""

    print("╔" + "═" * 68 + "╗")
    print("║" + "  DATE-BASED LOTTERY PATTERN DETECTION  ".center(68) + "║")
    print("║" + "  12 Methods Using Calendar Data  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    print("\n📊 LOADING DATA WITH DATES")
    draws, dates, df = load_data_with_dates(filepath, sep, date_col, number_cols, date_format)

    if number_cols is None:
        number_cols = [c for c in df.columns if c != date_col]

    results = {}

    results['d01'] = method_d01_day_of_week(draws, dates, max_number)
    results['d02'] = method_d02_monthly(draws, dates, max_number)
    results['d03'] = method_d03_spacing(draws, dates)
    results['d04'] = method_d04_date_number(draws, dates, max_number)
    results['d05'] = method_d05_trend(draws, dates)
    results['d06'] = method_d06_day_of_month(draws, dates, max_number)
    results['d07'] = method_d07_yearly(draws, dates, max_number)
    results['d08'] = method_d08_structural_break(draws, dates)
    results['d09'] = method_d09_weekday_weekend(draws, dates, max_number)
    results['d10'] = method_d10_lunar(draws, dates)
    results['d11'] = method_d11_multivariate(draws, dates, max_number)
    results['d12'] = method_d12_rolling_anomaly(draws, dates, max_number)

    print_date_summary(results)

    return results


if __name__ == "__main__":
    results = run_date_analysis(
        filepath='Data.csv',
        sep=';',
        date_col='date',
        number_cols=['n1', 'n2', 'n3', 'n4', 'n5'],
        max_number=49,
        date_format='%d/%m/%Y'
    )
