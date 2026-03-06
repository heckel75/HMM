"""
============================================================================
COMPREHENSIVE LOTTERY PATTERN DETECTION - BEYOND HMM
============================================================================

This pipeline tests for SUBTLE rigging patterns that frequency-based
methods and HMMs miss. Each method targets a different type of manipulation.

METHODS COVERED:
  1.  Frequency Analysis (Chi-squared, individual number bias)
  2.  Pair/Combination Analysis (co-occurrence patterns)
  3.  Triplet Analysis (3-number combinations)
  4.  Serial Correlation (autocorrelation at multiple lags)
  5.  Runs Test (sequential randomness)
  6.  Gap Analysis (intervals between appearances of each number)
  7.  Sum & Range Distribution (draw-level statistical control)
  8.  Spectral / Fourier Analysis (periodic rigging patterns)
  9.  Conditional Probability (does number X predict number Y next draw?)
  10. Mutual Information (non-linear dependencies between draws)
  11. Benford's Law (leading digit distribution)
  12. Odd/Even & High/Low Balance (structural balance manipulation)
  13. Consecutive Number Analysis (adjacent number frequency)
  14. Kolmogorov-Smirnov Test (distribution shape testing)
  15. Variance Ratio Test (over/under-dispersion detection)
  16. Recurrence Analysis (long-range repetition patterns)
  17. Machine Learning Predictability Test (can an ML model predict?)

============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency, entropy, ks_2samp, chisquare
from scipy.fft import fft, fftfreq
from itertools import combinations
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# DATA LOADING
# =====================================================================

def load_data(filepath='Data.csv', sep=';', columns=None):
    """Load lottery data from CSV."""
    df = pd.read_csv(filepath, sep=sep)
    if columns is None:
        columns = df.columns.tolist()
    draws = df[columns].values
    print(f"  Loaded {len(draws)} draws, {draws.shape[1]} numbers per draw")
    print(f"  Number range: {draws.min()} to {draws.max()}")
    print(f"  Columns: {columns}")
    return draws


# =====================================================================
# METHOD 1: FREQUENCY ANALYSIS
# =====================================================================

def method_01_frequency(draws, max_number=49):
    """
    WHAT IT TESTS:
      Are some numbers drawn more/less often than expected?
    
    HOW IT WORKS:
      Count how often each number appears across all draws.
      Compare to uniform expectation using Chi-squared test.
    
    WHAT RIGGING IT CATCHES:
      Direct number favoritism (e.g., ball weighting, machine bias).
    
    WHAT IT MISSES:
      Pair correlations, temporal patterns, structural manipulation.
    """
    print("\n" + "=" * 70)
    print("METHOD 1: FREQUENCY ANALYSIS (Chi-Squared)")
    print("=" * 70)
    
    all_numbers = draws.flatten()
    n_per_draw = draws.shape[1]
    n_draws = len(draws)
    
    observed = np.bincount(all_numbers, minlength=max_number + 1)[1:]
    expected = np.full(max_number, len(all_numbers) / max_number)
    
    chi2, p_value = chisquare(observed, expected)
    
    print(f"\n  Total numbers drawn: {len(all_numbers)}")
    print(f"  Expected per number: {expected[0]:.1f}")
    print(f"  Chi² statistic: {chi2:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Verdict: {'⚠️  NON-UNIFORM (p < 0.05)' if p_value < 0.05 else '✓ Uniform distribution'}")
    
    # Deviation analysis
    deviation = (observed - expected) / expected * 100
    sorted_idx = np.argsort(deviation)
    
    print(f"\n  Top 5 OVERREPRESENTED numbers:")
    for i in sorted_idx[-5:][::-1]:
        print(f"    Number {i+1:2d}: appeared {observed[i]:3d} times "
              f"(expected {expected[i]:.1f}, deviation {deviation[i]:+.1f}%)")
    
    print(f"\n  Top 5 UNDERREPRESENTED numbers:")
    for i in sorted_idx[:5]:
        print(f"    Number {i+1:2d}: appeared {observed[i]:3d} times "
              f"(expected {expected[i]:.1f}, deviation {deviation[i]:+.1f}%)")
    
    return {'chi2': chi2, 'p_value': p_value, 'deviation': deviation}


# =====================================================================
# METHOD 2: PAIR CO-OCCURRENCE ANALYSIS
# =====================================================================

def method_02_pairs(draws, max_number=49, n_simulations=10000):
    """
    WHAT IT TESTS:
      Do certain pairs of numbers appear together more/less than chance?
    
    HOW IT WORKS:
      Count how often each pair co-occurs in the same draw.
      Compare to expected frequency if draws were independent.
      Use Monte Carlo simulation to establish significance thresholds.
    
    WHAT RIGGING IT CATCHES:
      Mechanical linkages, correlated ball selection, combo manipulation.
    
    WHAT IT MISSES:
      Temporal patterns, single-number bias.
    """
    print("\n" + "=" * 70)
    print("METHOD 2: PAIR CO-OCCURRENCE ANALYSIS")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    
    # Count all pairs
    pair_counts = Counter()
    for draw in draws:
        for pair in combinations(sorted(draw), 2):
            pair_counts[pair] += 1
    
    # Expected pair frequency under independence
    # P(both i and j in a draw) = C(47,3)/C(49,5) for 5-from-49
    # Simpler: use individual frequencies
    number_freq = np.bincount(draws.flatten(), minlength=max_number + 1)[1:] / n_draws
    
    # Calculate expected counts and chi-squared for pairs
    pair_deviations = {}
    for pair, observed_count in pair_counts.items():
        i, j = pair
        # Expected co-occurrence under independence
        p_i = number_freq[i - 1]
        p_j = number_freq[j - 1]
        # Approximate expected count
        expected_count = p_i * p_j * n_draws * (n_per_draw * (n_per_draw - 1)) / (max_number * (max_number - 1) / (n_draws * n_per_draw / n_draws))
        
        # Simpler: expected count based on hypergeometric
        expected_count = n_draws * (n_per_draw / max_number) * ((n_per_draw - 1) / (max_number - 1))
        
        deviation = (observed_count - expected_count) / expected_count * 100 if expected_count > 0 else 0
        pair_deviations[pair] = {
            'observed': observed_count,
            'expected': expected_count,
            'deviation_pct': deviation
        }
    
    # Sort by absolute deviation
    sorted_pairs = sorted(pair_deviations.items(), 
                          key=lambda x: abs(x[1]['deviation_pct']), reverse=True)
    
    print(f"\n  Total unique pairs observed: {len(pair_counts)}")
    print(f"  Expected co-occurrence per pair: {sorted_pairs[0][1]['expected']:.2f}")
    
    print(f"\n  Top 10 MOST SUSPICIOUS PAIRS (highest deviation):")
    for pair, info in sorted_pairs[:10]:
        print(f"    ({pair[0]:2d}, {pair[1]:2d}): observed {info['observed']:3d}, "
              f"expected {info['expected']:.1f}, deviation {info['deviation_pct']:+.1f}%")
    
    print(f"\n  Top 10 MOST AVOIDED PAIRS (lowest deviation):")
    for pair, info in sorted_pairs[-10:][::-1]:
        print(f"    ({pair[0]:2d}, {pair[1]:2d}): observed {info['observed']:3d}, "
              f"expected {info['expected']:.1f}, deviation {info['deviation_pct']:+.1f}%")
    
    # Overall pair uniformity test
    all_pair_obs = [v['observed'] for v in pair_deviations.values()]
    all_pair_exp = [v['expected'] for v in pair_deviations.values()]
    
    if len(set(all_pair_exp)) == 1:  # All same expected
        chi2, p = chisquare(all_pair_obs)
    else:
        chi2, p = chisquare(all_pair_obs, all_pair_exp)
    
    print(f"\n  Pair uniformity Chi²: {chi2:.2f}")
    print(f"  p-value: {p:.6f}")
    print(f"  Verdict: {'⚠️  PAIR PATTERNS DETECTED' if p < 0.05 else '✓ Pairs look random'}")
    
    return {'pair_deviations': pair_deviations, 'chi2': chi2, 'p_value': p}


# =====================================================================
# METHOD 3: TRIPLET ANALYSIS
# =====================================================================

def method_03_triplets(draws, max_number=49):
    """
    WHAT IT TESTS:
      Do certain 3-number combinations repeat more than expected?
    
    HOW IT WORKS:
      Count all triplet occurrences across draws.
      In a fair lottery with ~1000 draws, triplet repeats should be rare.
    
    WHAT RIGGING IT CATCHES:
      Pre-determined draw sequences, combination recycling.
    """
    print("\n" + "=" * 70)
    print("METHOD 3: TRIPLET REPEAT ANALYSIS")
    print("=" * 70)
    
    triplet_counts = Counter()
    for draw in draws:
        for triplet in combinations(sorted(draw), 3):
            triplet_counts[triplet] += 1
    
    # Count repeats
    repeats = {k: v for k, v in triplet_counts.items() if v > 1}
    max_repeat = max(triplet_counts.values()) if triplet_counts else 0
    
    total_triplets = len(triplet_counts)
    total_possible = len(list(combinations(range(1, max_number + 1), 3)))
    
    print(f"\n  Unique triplets observed: {total_triplets}")
    print(f"  Total possible triplets: {total_possible}")
    print(f"  Coverage: {total_triplets/total_possible*100:.2f}%")
    print(f"  Triplets appearing more than once: {len(repeats)}")
    print(f"  Maximum repeat count: {max_repeat}")
    
    if repeats:
        sorted_repeats = sorted(repeats.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top repeated triplets:")
        for triplet, count in sorted_repeats[:15]:
            print(f"    {triplet}: appeared {count} times")
    
    # Monte Carlo: how many repeats expected in random data?
    print(f"\n  Running Monte Carlo simulation (1000 iterations)...")
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    mc_repeats = []
    for _ in range(1000):
        sim_draws = np.array([np.sort(np.random.choice(range(1, max_number + 1), 
                              size=n_per_draw, replace=False)) for _ in range(n_draws)])
        sim_counts = Counter()
        for draw in sim_draws:
            for triplet in combinations(draw, 3):
                sim_counts[triplet] += 1
        mc_repeats.append(sum(1 for v in sim_counts.values() if v > 1))
    
    mc_mean = np.mean(mc_repeats)
    mc_std = np.std(mc_repeats)
    z_score = (len(repeats) - mc_mean) / mc_std if mc_std > 0 else 0
    
    print(f"  Simulated repeats: mean={mc_mean:.1f}, std={mc_std:.1f}")
    print(f"  Observed repeats: {len(repeats)}")
    print(f"  Z-score: {z_score:.2f}")
    print(f"  Verdict: {'⚠️  ABNORMAL TRIPLET REPEATS' if abs(z_score) > 2 else '✓ Triplet repeats look normal'}")
    
    return {'n_repeats': len(repeats), 'max_repeat': max_repeat, 'z_score': z_score}


# =====================================================================
# METHOD 4: SERIAL AUTOCORRELATION
# =====================================================================

def method_04_autocorrelation(draws, max_lag=30):
    """
    WHAT IT TESTS:
      Are consecutive draws correlated? Does draw N predict draw N+1?
    
    HOW IT WORKS:
      Compute autocorrelation of the draw sum at lags 1 to max_lag.
      Also test autocorrelation of individual number presence.
    
    WHAT RIGGING IT CATCHES:
      Sequential dependencies, alternating patterns, memory effects.
    """
    print("\n" + "=" * 70)
    print("METHOD 4: SERIAL AUTOCORRELATION")
    print("=" * 70)
    
    draw_sums = draws.sum(axis=1)
    n = len(draw_sums)
    confidence = 2 / np.sqrt(n)
    
    print(f"\n  Testing lags 1-{max_lag}")
    print(f"  95% confidence band: ±{confidence:.4f}")
    
    significant_lags = []
    autocorrs = []
    
    for lag in range(1, max_lag + 1):
        corr = np.corrcoef(draw_sums[:-lag], draw_sums[lag:])[0, 1]
        autocorrs.append(corr)
        is_sig = abs(corr) > confidence
        if is_sig:
            significant_lags.append(lag)
            print(f"  ⚠️  Lag {lag:2d}: r = {corr:+.4f} (SIGNIFICANT)")
    
    n_significant = len(significant_lags)
    expected_false_positives = max_lag * 0.05
    
    print(f"\n  Significant lags: {n_significant} out of {max_lag}")
    print(f"  Expected by chance (5%): {expected_false_positives:.1f}")
    
    # Also test per-number autocorrelation
    print(f"\n  Per-number autocorrelation (lag 1):")
    max_number = draws.max()
    per_number_sigs = 0
    for num in range(1, max_number + 1):
        presence = np.array([1 if num in draw else 0 for draw in draws])
        if presence.sum() > 5:  # Need enough data
            corr = np.corrcoef(presence[:-1], presence[1:])[0, 1]
            if abs(corr) > confidence:
                per_number_sigs += 1
                print(f"    ⚠️  Number {num:2d}: lag-1 autocorr = {corr:+.4f}")
    
    print(f"\n  Numbers with significant lag-1 autocorrelation: {per_number_sigs}")
    
    verdict_suspicious = n_significant > expected_false_positives * 2 or per_number_sigs > max_number * 0.1
    print(f"  Verdict: {'⚠️  SERIAL DEPENDENCIES DETECTED' if verdict_suspicious else '✓ No serial correlation'}")
    
    return {'autocorrs': autocorrs, 'significant_lags': significant_lags, 
            'per_number_sigs': per_number_sigs}


# =====================================================================
# METHOD 5: RUNS TEST
# =====================================================================

def method_05_runs(draws):
    """
    WHAT IT TESTS:
      Is the sequence of "above/below median" draws truly random?
    
    HOW IT WORKS:
      Convert draw sums to binary (above/below median).
      Count "runs" (consecutive sequences of same type).
      Too few runs = clustering. Too many runs = alternating.
    
    WHAT RIGGING IT CATCHES:
      Clustering of high/low draws, forced alternation patterns.
    """
    print("\n" + "=" * 70)
    print("METHOD 5: RUNS TEST")
    print("=" * 70)
    
    draw_sums = draws.sum(axis=1)
    median_val = np.median(draw_sums)
    binary = (draw_sums > median_val).astype(int)
    
    # Count runs
    runs = 1 + np.sum(np.diff(binary) != 0)
    n1 = np.sum(binary == 1)
    n0 = np.sum(binary == 0)
    n = n0 + n1
    
    expected_runs = 1 + 2 * n0 * n1 / n
    var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n**2 * (n - 1))
    std_runs = np.sqrt(var_runs)
    z = (runs - expected_runs) / std_runs
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    print(f"\n  Median draw sum: {median_val:.1f}")
    print(f"  Above median: {n1}, Below median: {n0}")
    print(f"  Observed runs: {runs}")
    print(f"  Expected runs: {expected_runs:.1f}")
    print(f"  Std deviation: {std_runs:.2f}")
    print(f"  Z-score: {z:.4f}")
    print(f"  p-value: {p_value:.6f}")
    
    if z < -2:
        interpretation = "⚠️  TOO FEW RUNS → clustering detected"
    elif z > 2:
        interpretation = "⚠️  TOO MANY RUNS → forced alternation detected"
    else:
        interpretation = "✓ Random sequence"
    print(f"  Verdict: {interpretation}")
    
    # Also test runs for each number
    print(f"\n  Per-number runs test (numbers with p < 0.05):")
    max_number = draws.max()
    suspicious_numbers = []
    for num in range(1, max_number + 1):
        presence = np.array([1 if num in draw else 0 for draw in draws])
        r = 1 + np.sum(np.diff(presence) != 0)
        m1 = presence.sum()
        m0 = len(presence) - m1
        if m1 > 0 and m0 > 0:
            exp_r = 1 + 2 * m0 * m1 / (m0 + m1)
            var_r = (2 * m0 * m1 * (2 * m0 * m1 - m0 - m1)) / ((m0 + m1)**2 * (m0 + m1 - 1))
            if var_r > 0:
                z_num = (r - exp_r) / np.sqrt(var_r)
                p_num = 2 * (1 - stats.norm.cdf(abs(z_num)))
                if p_num < 0.05:
                    suspicious_numbers.append((num, z_num, p_num))
                    print(f"    Number {num:2d}: Z={z_num:+.3f}, p={p_num:.4f}")
    
    print(f"\n  Numbers with suspicious run patterns: {len(suspicious_numbers)}/{max_number}")
    
    return {'z': z, 'p_value': p_value, 'suspicious_numbers': suspicious_numbers}


# =====================================================================
# METHOD 6: GAP ANALYSIS
# =====================================================================

def method_06_gaps(draws, max_number=49):
    """
    WHAT IT TESTS:
      Is the gap between consecutive appearances of each number normal?
    
    HOW IT WORKS:
      For each number, record the gaps (draws between appearances).
      In a fair lottery, gaps follow a geometric distribution.
      Test if observed gap distribution matches expected.
    
    WHAT RIGGING IT CATCHES:
      Numbers being "due" (artificially shortened gaps), 
      numbers being suppressed (artificially long gaps),
      periodic number injection.
    """
    print("\n" + "=" * 70)
    print("METHOD 6: GAP ANALYSIS")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    p_appear = n_per_draw / max_number  # Expected probability of appearing
    
    print(f"\n  Expected appearance probability: {p_appear:.4f}")
    print(f"  Expected mean gap: {1/p_appear:.1f} draws")
    
    suspicious_numbers = []
    all_gaps = []
    
    for num in range(1, max_number + 1):
        appearances = [i for i, draw in enumerate(draws) if num in draw]
        if len(appearances) < 2:
            continue
        
        gaps = np.diff(appearances)
        all_gaps.extend(gaps)
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        max_gap = np.max(gaps)
        min_gap = np.min(gaps)
        
        # Expected geometric distribution: mean = 1/p, var = (1-p)/p²
        expected_mean = 1 / p_appear
        expected_std = np.sqrt((1 - p_appear) / p_appear**2)
        
        # Z-test for mean gap
        se = expected_std / np.sqrt(len(gaps))
        z = (mean_gap - expected_mean) / se if se > 0 else 0
        
        if abs(z) > 2:
            suspicious_numbers.append((num, mean_gap, z))
            print(f"  ⚠️  Number {num:2d}: mean gap={mean_gap:.1f} (expected {expected_mean:.1f}), "
                  f"Z={z:+.2f}, range=[{min_gap},{max_gap}]")
    
    # Test overall gap distribution against geometric
    all_gaps = np.array(all_gaps)
    print(f"\n  Overall gap statistics:")
    print(f"  Mean gap: {np.mean(all_gaps):.2f} (expected: {1/p_appear:.2f})")
    print(f"  Std gap:  {np.std(all_gaps):.2f}")
    print(f"  Max gap:  {np.max(all_gaps)}")
    print(f"  Min gap:  {np.min(all_gaps)}")
    
    # KS test against geometric distribution
    from scipy.stats import geom
    theoretical_gaps = geom.rvs(p_appear, size=len(all_gaps))
    ks_stat, ks_p = ks_2samp(all_gaps, theoretical_gaps)
    
    print(f"\n  KS test (observed vs geometric): statistic={ks_stat:.4f}, p={ks_p:.6f}")
    print(f"  Numbers with abnormal gap patterns: {len(suspicious_numbers)}")
    print(f"  Verdict: {'⚠️  GAP ANOMALIES DETECTED' if ks_p < 0.05 or len(suspicious_numbers) > max_number * 0.1 else '✓ Gaps look normal'}")
    
    return {'suspicious_numbers': suspicious_numbers, 'ks_p': ks_p, 'all_gaps': all_gaps}


# =====================================================================
# METHOD 7: SUM & RANGE DISTRIBUTION
# =====================================================================

def method_07_sum_range(draws, max_number=49):
    """
    WHAT IT TESTS:
      Are draw sums and ranges distributed as expected?
    
    HOW IT WORKS:
      Compute sum and range of each draw.
      Compare distribution to Monte Carlo simulation of fair draws.
    
    WHAT RIGGING IT CATCHES:
      Draws being constrained to "look normal" (controlled sums),
      avoiding extreme draws, variance manipulation.
    """
    print("\n" + "=" * 70)
    print("METHOD 7: SUM & RANGE DISTRIBUTION")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    
    obs_sums = draws.sum(axis=1)
    obs_ranges = draws.max(axis=1) - draws.min(axis=1)
    obs_stds = draws.std(axis=1)
    
    # Monte Carlo reference
    print(f"\n  Generating Monte Carlo reference (5000 simulations)...")
    mc_sums = []
    mc_ranges = []
    mc_stds = []
    for _ in range(5000):
        sim = np.sort(np.random.choice(range(1, max_number + 1), 
                      size=n_per_draw, replace=False))
        mc_sums.append(sim.sum())
        mc_ranges.append(sim.max() - sim.min())
        mc_stds.append(sim.std())
    
    mc_sums = np.array(mc_sums)
    mc_ranges = np.array(mc_ranges)
    mc_stds = np.array(mc_stds)
    
    # Compare distributions
    for name, obs, mc in [("Sum", obs_sums, mc_sums), 
                           ("Range", obs_ranges, mc_ranges),
                           ("Std Dev", obs_stds, mc_stds)]:
        ks_stat, ks_p = ks_2samp(obs, mc)
        print(f"\n  {name}:")
        print(f"    Observed: mean={np.mean(obs):.2f}, std={np.std(obs):.2f}")
        print(f"    Expected: mean={np.mean(mc):.2f}, std={np.std(mc):.2f}")
        print(f"    KS test: statistic={ks_stat:.4f}, p={ks_p:.6f}")
        print(f"    {'⚠️  SIGNIFICANT DIFFERENCE' if ks_p < 0.05 else '✓ Matches expected'}")
    
    # Check if variance of sums is too low (over-controlled) or too high
    var_ratio = np.var(obs_sums) / np.var(mc_sums)
    print(f"\n  Variance ratio (observed/expected) for sums: {var_ratio:.4f}")
    print(f"  (< 0.85 suggests over-control, > 1.15 suggests excess variance)")
    
    return {'var_ratio': var_ratio}


# =====================================================================
# METHOD 8: SPECTRAL / FOURIER ANALYSIS
# =====================================================================

def method_08_spectral(draws, max_number=49):
    """
    WHAT IT TESTS:
      Are there periodic/cyclical patterns in the draws?
    
    HOW IT WORKS:
      Apply FFT to draw sum time series.
      Look for frequency peaks above noise floor.
    
    WHAT RIGGING IT CATCHES:
      Periodic rigging (every N draws), cyclical manipulation,
      scheduled interference patterns.
    """
    print("\n" + "=" * 70)
    print("METHOD 8: SPECTRAL / FOURIER ANALYSIS")
    print("=" * 70)
    
    draw_sums = draws.sum(axis=1)
    n = len(draw_sums)
    
    # Detrend
    draw_sums_detrended = draw_sums - np.mean(draw_sums)
    
    # FFT
    yf = fft(draw_sums_detrended)
    xf = fftfreq(n, d=1)
    
    # Power spectrum (only positive frequencies)
    positive_mask = xf > 0
    frequencies = xf[positive_mask]
    power = 2.0 / n * np.abs(yf[positive_mask])**2
    
    # Normalize
    power_normalized = power / np.mean(power)
    
    # Find peaks above threshold (e.g., 5x average power)
    threshold = 5.0
    peak_indices = np.where(power_normalized > threshold)[0]
    
    print(f"\n  Analyzed {n} draws")
    print(f"  Frequency resolution: {1/n:.4f} cycles/draw")
    print(f"  Peak detection threshold: {threshold}x mean power")
    
    if len(peak_indices) > 0:
        print(f"\n  ⚠️  {len(peak_indices)} significant frequency peaks found:")
        for idx in peak_indices[:10]:
            freq = frequencies[idx]
            period = 1 / freq if freq > 0 else float('inf')
            pwr = power_normalized[idx]
            print(f"    Frequency={freq:.4f} → Period={period:.1f} draws, "
                  f"Power={pwr:.1f}x average")
    else:
        print(f"\n  No significant frequency peaks found.")
    
    # Also test individual numbers for periodicity
    print(f"\n  Per-number periodicity check:")
    periodic_numbers = []
    for num in range(1, max_number + 1):
        presence = np.array([1.0 if num in draw else 0.0 for draw in draws])
        presence -= presence.mean()
        yf_num = fft(presence)
        power_num = 2.0 / n * np.abs(yf_num[positive_mask])**2
        power_num_norm = power_num / np.mean(power_num) if np.mean(power_num) > 0 else power_num
        max_power = np.max(power_num_norm) if len(power_num_norm) > 0 else 0
        
        if max_power > 8.0:  # Strong periodicity
            peak_freq = frequencies[np.argmax(power_num_norm)]
            period = 1 / peak_freq if peak_freq > 0 else float('inf')
            periodic_numbers.append((num, period, max_power))
            print(f"    ⚠️  Number {num:2d}: period={period:.1f} draws, power={max_power:.1f}x")
    
    print(f"\n  Numbers with periodic patterns: {len(periodic_numbers)}")
    print(f"  Verdict: {'⚠️  PERIODIC PATTERNS DETECTED' if len(peak_indices) > 2 or len(periodic_numbers) > 3 else '✓ No periodicity found'}")
    
    return {'peak_indices': peak_indices, 'periodic_numbers': periodic_numbers,
            'frequencies': frequencies, 'power': power_normalized}


# =====================================================================
# METHOD 9: CONDITIONAL PROBABILITY ANALYSIS
# =====================================================================

def method_09_conditional(draws, max_number=49):
    """
    WHAT IT TESTS:
      Does the presence of number X in draw N predict number Y in draw N+1?
    
    HOW IT WORKS:
      Build a conditional probability matrix P(Y in draw N+1 | X in draw N).
      Compare to unconditional probability P(Y appears).
    
    WHAT RIGGING IT CATCHES:
      Sequential dependencies, "trigger" numbers that influence next draw.
    """
    print("\n" + "=" * 70)
    print("METHOD 9: CONDITIONAL PROBABILITY (Cross-draw)")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    base_prob = n_per_draw / max_number
    
    # Build conditional probability matrix
    cond_matrix = np.zeros((max_number, max_number))
    cond_counts = np.zeros((max_number, max_number))
    presence_counts = np.zeros(max_number)
    
    for i in range(n_draws - 1):
        current_draw = set(draws[i])
        next_draw = set(draws[i + 1])
        
        for x in current_draw:
            presence_counts[x - 1] += 1
            for y in next_draw:
                cond_counts[x - 1][y - 1] += 1
    
    # Calculate conditional probabilities
    for x in range(max_number):
        if presence_counts[x] > 0:
            cond_matrix[x] = cond_counts[x] / presence_counts[x]
    
    # Find strongest dependencies
    deviations = cond_matrix - base_prob
    significant_pairs = []
    
    # Use binomial test for significance
    for x in range(max_number):
        for y in range(max_number):
            n_trials = int(presence_counts[x])
            if n_trials < 10:
                continue
            n_successes = int(cond_counts[x][y])
            p_val = stats.binomtest(n_successes, n_trials, base_prob).pvalue
            if p_val < 0.01:  # Strict threshold
                significant_pairs.append((x + 1, y + 1, cond_matrix[x][y], base_prob, p_val))
    
    print(f"\n  Base probability of any number: {base_prob:.4f}")
    print(f"  Significant conditional dependencies (p < 0.01): {len(significant_pairs)}")
    
    # Expected false positives at p < 0.01 with max_number² tests
    expected_fp = max_number * max_number * 0.01
    print(f"  Expected false positives: {expected_fp:.0f}")
    
    if significant_pairs:
        sorted_pairs = sorted(significant_pairs, key=lambda x: x[4])[:15]
        print(f"\n  Strongest conditional dependencies:")
        for x, y, cond_p, base_p, p in sorted_pairs:
            direction = "↑" if cond_p > base_p else "↓"
            print(f"    If {x:2d} drawn → P({y:2d} next) = {cond_p:.4f} "
                  f"(base: {base_p:.4f}, {direction}, p={p:.6f})")
    
    suspicious = len(significant_pairs) > expected_fp * 2
    print(f"\n  Verdict: {'⚠️  CROSS-DRAW DEPENDENCIES FOUND' if suspicious else '✓ No conditional dependencies'}")
    
    return {'n_significant': len(significant_pairs), 'significant_pairs': significant_pairs}


# =====================================================================
# METHOD 10: MUTUAL INFORMATION
# =====================================================================

def method_10_mutual_information(draws, max_number=49):
    """
    WHAT IT TESTS:
      Is there non-linear dependency between consecutive draws?
    
    HOW IT WORKS:
      Discretize draw features and compute mutual information
      between draw N and draw N+1. MI > 0 means information leaks
      between draws (non-independence).
    
    WHAT RIGGING IT CATCHES:
      Any form of dependency, including non-linear ones that
      correlation tests miss.
    """
    print("\n" + "=" * 70)
    print("METHOD 10: MUTUAL INFORMATION ANALYSIS")
    print("=" * 70)
    
    draw_sums = draws.sum(axis=1)
    
    # Discretize into bins
    n_bins = 10
    bins = np.linspace(draw_sums.min(), draw_sums.max() + 1, n_bins + 1)
    digitized = np.digitize(draw_sums, bins)
    
    # Compute MI between consecutive draws
    current = digitized[:-1]
    next_val = digitized[1:]
    
    # Joint distribution
    joint_counts = np.zeros((n_bins + 1, n_bins + 1))
    for c, n in zip(current, next_val):
        joint_counts[c][n] += 1
    
    joint_prob = joint_counts / joint_counts.sum()
    marginal_current = joint_prob.sum(axis=1)
    marginal_next = joint_prob.sum(axis=0)
    
    mi = 0
    for i in range(n_bins + 1):
        for j in range(n_bins + 1):
            if joint_prob[i][j] > 0 and marginal_current[i] > 0 and marginal_next[j] > 0:
                mi += joint_prob[i][j] * np.log2(joint_prob[i][j] / 
                      (marginal_current[i] * marginal_next[j]))
    
    # Permutation test for significance
    print(f"\n  Mutual Information: {mi:.6f} bits")
    print(f"  Running permutation test (1000 shuffles)...")
    
    mi_null = []
    for _ in range(1000):
        shuffled = np.random.permutation(digitized)
        curr_s = shuffled[:-1]
        next_s = shuffled[1:]
        
        jc = np.zeros((n_bins + 1, n_bins + 1))
        for c, n in zip(curr_s, next_s):
            jc[c][n] += 1
        jp = jc / jc.sum()
        mc = jp.sum(axis=1)
        mn = jp.sum(axis=0)
        
        mi_s = 0
        for i in range(n_bins + 1):
            for j in range(n_bins + 1):
                if jp[i][j] > 0 and mc[i] > 0 and mn[j] > 0:
                    mi_s += jp[i][j] * np.log2(jp[i][j] / (mc[i] * mn[j]))
        mi_null.append(mi_s)
    
    mi_null = np.array(mi_null)
    p_value = np.mean(mi_null >= mi)
    
    print(f"  Null MI: mean={np.mean(mi_null):.6f}, std={np.std(mi_null):.6f}")
    print(f"  Observed MI: {mi:.6f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Verdict: {'⚠️  NON-LINEAR DEPENDENCY DETECTED' if p_value < 0.05 else '✓ No mutual information leakage'}")
    
    # Also test MI at larger lags
    print(f"\n  MI at multiple lags:")
    for lag in [2, 3, 5, 10]:
        curr_l = digitized[:-lag]
        next_l = digitized[lag:]
        jc = np.zeros((n_bins + 1, n_bins + 1))
        for c, n in zip(curr_l, next_l):
            jc[c][n] += 1
        jp = jc / jc.sum()
        mc = jp.sum(axis=1)
        mn = jp.sum(axis=0)
        mi_l = 0
        for i in range(n_bins + 1):
            for j in range(n_bins + 1):
                if jp[i][j] > 0 and mc[i] > 0 and mn[j] > 0:
                    mi_l += jp[i][j] * np.log2(jp[i][j] / (mc[i] * mn[j]))
        print(f"    Lag {lag:2d}: MI = {mi_l:.6f}")
    
    return {'mi': mi, 'p_value': p_value}


# =====================================================================
# METHOD 11: BENFORD'S LAW
# =====================================================================

def method_11_benford(draws):
    """
    WHAT IT TESTS:
      Do leading digits follow Benford's Law?
    
    HOW IT WORKS:
      Benford's Law predicts the distribution of leading digits in 
      naturally occurring datasets. Lottery numbers don't strictly
      follow Benford's, but deviations can reveal manipulation.
    
    WHAT RIGGING IT CATCHES:
      Human-generated numbers (people are bad at faking randomness),
      algorithmic manipulation that doesn't account for digit distribution.
    
    NOTE:
      This test is less applicable to bounded lottery numbers but
      included for completeness. More useful for draw sums.
    """
    print("\n" + "=" * 70)
    print("METHOD 11: BENFORD'S LAW ANALYSIS")
    print("=" * 70)
    
    # Apply to draw sums (more applicable than individual numbers)
    draw_sums = draws.sum(axis=1)
    
    # Extract leading digits
    leading_digits = []
    for s in draw_sums:
        leading_digits.append(int(str(int(s))[0]))
    
    observed = np.zeros(9)
    for d in leading_digits:
        if 1 <= d <= 9:
            observed[d - 1] += 1
    
    # Benford's expected distribution
    benford_expected = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
    expected_counts = benford_expected * len(leading_digits)
    
    chi2, p_value = chisquare(observed, expected_counts)
    
    print(f"\n  Applied to draw sums (range: {draw_sums.min()}-{draw_sums.max()})")
    print(f"\n  Digit  Observed  Expected  Deviation")
    print(f"  {'─' * 45}")
    for d in range(9):
        dev = (observed[d] - expected_counts[d]) / expected_counts[d] * 100
        print(f"    {d+1}     {observed[d]:6.0f}    {expected_counts[d]:6.1f}    {dev:+.1f}%")
    
    print(f"\n  Chi² = {chi2:.4f}, p-value = {p_value:.6f}")
    print(f"  Verdict: {'⚠️  DEVIATES FROM BENFORD' if p_value < 0.05 else '✓ Consistent with Benford'}")
    print(f"  (Note: Bounded lottery sums don't perfectly follow Benford's Law,")
    print(f"   so this test is supplementary, not definitive)")
    
    return {'chi2': chi2, 'p_value': p_value}


# =====================================================================
# METHOD 12: ODD/EVEN & HIGH/LOW BALANCE
# =====================================================================

def method_12_balance(draws, max_number=49):
    """
    WHAT IT TESTS:
      Are odd/even and high/low number ratios natural?
    
    HOW IT WORKS:
      Count odd/even and high/low splits in each draw.
      Compare distribution to expected binomial distribution.
    
    WHAT RIGGING IT CATCHES:
      Balance manipulation (forcing "aesthetic" draws),
      avoiding all-odd, all-even, all-high, all-low draws.
    """
    print("\n" + "=" * 70)
    print("METHOD 12: ODD/EVEN & HIGH/LOW BALANCE")
    print("=" * 70)
    
    n_per_draw = draws.shape[1]
    mid_point = max_number / 2
    
    # Odd/Even counts per draw
    even_counts = [np.sum(draw % 2 == 0) for draw in draws]
    high_counts = [np.sum(draw > mid_point) for draw in draws]
    
    # Expected distribution (hypergeometric)
    n_even = max_number // 2
    n_odd = max_number - n_even
    
    for name, counts, n_type in [("Even numbers", even_counts, n_even),
                                   ("High numbers", high_counts, max_number // 2)]:
        counter = Counter(counts)
        print(f"\n  {name} per draw:")
        print(f"  Count | Observed | Expected (hypergeometric)")
        print(f"  {'─' * 45}")
        
        from scipy.stats import hypergeom
        expected_dist = {}
        for k in range(n_per_draw + 1):
            p = hypergeom.pmf(k, max_number, n_type, n_per_draw)
            expected_dist[k] = p * len(draws)
        
        obs_list = []
        exp_list = []
        for k in range(n_per_draw + 1):
            obs = counter.get(k, 0)
            exp = expected_dist.get(k, 0)
            obs_list.append(obs)
            exp_list.append(exp)
            if exp > 0:
                print(f"    {k:3d}  |  {obs:5d}   |  {exp:7.1f}")
        
        chi2, p = chisquare([o for o, e in zip(obs_list, exp_list) if e > 1],
                            [e for e in exp_list if e > 1])
        print(f"  Chi² = {chi2:.4f}, p = {p:.6f}")
        print(f"  {'⚠️  IMBALANCED' if p < 0.05 else '✓ Normal balance'}")
    
    return {}


# =====================================================================
# METHOD 13: CONSECUTIVE NUMBER ANALYSIS
# =====================================================================

def method_13_consecutive(draws, max_number=49):
    """
    WHAT IT TESTS:
      Do consecutive numbers (e.g., 14-15, 32-33) appear together
      more or less than expected?
    
    HOW IT WORKS:
      Count draws containing at least one consecutive pair.
      Compare to Monte Carlo expected rate.
    
    WHAT RIGGING IT CATCHES:
      Mechanical adjacency bias, sequential selection patterns.
    """
    print("\n" + "=" * 70)
    print("METHOD 13: CONSECUTIVE NUMBER ANALYSIS")
    print("=" * 70)
    
    n_per_draw = draws.shape[1]
    
    # Count consecutive pairs in each draw
    consec_per_draw = []
    for draw in draws:
        sorted_draw = np.sort(draw)
        n_consec = np.sum(np.diff(sorted_draw) == 1)
        consec_per_draw.append(n_consec)
    
    consec_per_draw = np.array(consec_per_draw)
    draws_with_consec = np.sum(consec_per_draw > 0)
    
    # Monte Carlo reference
    mc_consec_rates = []
    for _ in range(5000):
        sim = np.sort(np.random.choice(range(1, max_number + 1), 
                      size=n_per_draw, replace=False))
        mc_consec_rates.append(np.sum(np.diff(sim) == 1) > 0)
    
    expected_rate = np.mean(mc_consec_rates)
    observed_rate = draws_with_consec / len(draws)
    
    # Binomial test
    p_value = stats.binomtest(draws_with_consec, len(draws), expected_rate).pvalue
    
    print(f"\n  Draws with consecutive numbers: {draws_with_consec}/{len(draws)} ({observed_rate:.1%})")
    print(f"  Expected rate: {expected_rate:.1%}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Verdict: {'⚠️  ABNORMAL CONSECUTIVE RATE' if p_value < 0.05 else '✓ Normal consecutive rate'}")
    
    # Distribution of consecutive count
    print(f"\n  Distribution of consecutive pairs per draw:")
    for n_c in range(max(consec_per_draw) + 1):
        count = np.sum(consec_per_draw == n_c)
        print(f"    {n_c} consecutive pairs: {count} draws ({count/len(draws)*100:.1f}%)")
    
    return {'observed_rate': observed_rate, 'expected_rate': expected_rate, 'p_value': p_value}


# =====================================================================
# METHOD 14: KOLMOGOROV-SMIRNOV TEST
# =====================================================================

def method_14_ks_test(draws, max_number=49):
    """
    WHAT IT TESTS:
      Does the overall number distribution match uniform?
    
    HOW IT WORKS:
      KS test compares the empirical CDF of drawn numbers
      to the theoretical uniform CDF. More sensitive to
      distributional shape differences than chi-squared.
    
    WHAT RIGGING IT CATCHES:
      Subtle distributional skew that chi-squared might miss.
    """
    print("\n" + "=" * 70)
    print("METHOD 14: KOLMOGOROV-SMIRNOV DISTRIBUTION TEST")
    print("=" * 70)
    
    all_numbers = draws.flatten()
    
    # KS test against uniform distribution
    ks_stat, ks_p = stats.kstest(all_numbers, 'uniform', 
                                  args=(0.5, max_number))
    
    print(f"\n  KS statistic: {ks_stat:.6f}")
    print(f"  p-value: {ks_p:.6f}")
    print(f"  Verdict: {'⚠️  DISTRIBUTION NOT UNIFORM' if ks_p < 0.05 else '✓ Distribution is uniform'}")
    
    # Also test first half vs second half of draws
    mid = len(draws) // 2
    first_half = draws[:mid].flatten()
    second_half = draws[mid:].flatten()
    ks_halves, p_halves = ks_2samp(first_half, second_half)
    
    print(f"\n  First half vs second half:")
    print(f"  KS statistic: {ks_halves:.6f}")
    print(f"  p-value: {p_halves:.6f}")
    print(f"  Verdict: {'⚠️  DISTRIBUTION CHANGED OVER TIME' if p_halves < 0.05 else '✓ Consistent across time'}")
    
    # Sliding window analysis
    print(f"\n  Sliding window analysis (window=100 draws):")
    window_size = 100
    if len(draws) >= window_size * 2:
        changes_detected = 0
        for start in range(0, len(draws) - window_size * 2, window_size // 2):
            w1 = draws[start:start + window_size].flatten()
            w2 = draws[start + window_size:start + window_size * 2].flatten()
            _, p = ks_2samp(w1, w2)
            if p < 0.01:
                changes_detected += 1
                print(f"    ⚠️  Draws {start}-{start+window_size} vs "
                      f"{start+window_size}-{start+window_size*2}: p={p:.4f}")
        
        if changes_detected == 0:
            print(f"    No significant distribution changes detected")
    
    return {'ks_stat': ks_stat, 'ks_p': ks_p}


# =====================================================================
# METHOD 15: VARIANCE RATIO TEST
# =====================================================================

def method_15_variance_ratio(draws):
    """
    WHAT IT TESTS:
      Is the variance of draw sums consistent over time?
      
    HOW IT WORKS:
      Split the data into blocks and compare variance across blocks.
      Also test if variance is too low (over-controlled) or too high.
    
    WHAT RIGGING IT CATCHES:
      Periods of controlled draws (low variance), periods of
      "letting it run" (high variance), variance regime shifts.
    """
    print("\n" + "=" * 70)
    print("METHOD 15: VARIANCE RATIO TEST")
    print("=" * 70)
    
    draw_sums = draws.sum(axis=1)
    n = len(draw_sums)
    
    # Split into blocks
    block_size = 50
    n_blocks = n // block_size
    
    block_variances = []
    block_means = []
    for i in range(n_blocks):
        block = draw_sums[i * block_size:(i + 1) * block_size]
        block_variances.append(np.var(block))
        block_means.append(np.mean(block))
    
    # Bartlett's test for homogeneity of variances
    blocks = [draw_sums[i * block_size:(i + 1) * block_size].astype(float) for i in range(n_blocks)]
    # Filter out blocks with zero variance
    blocks = [b for b in blocks if np.var(b) > 0]
    try:
        bartlett_stat, bartlett_p = stats.bartlett(*blocks)
    except Exception:
        bartlett_stat, bartlett_p = 0.0, 1.0
    
    print(f"\n  Block size: {block_size} draws, {n_blocks} blocks")
    print(f"  Overall variance: {np.var(draw_sums):.2f}")
    print(f"  Block variance range: [{min(block_variances):.1f}, {max(block_variances):.1f}]")
    print(f"  Coefficient of variation of block variances: {np.std(block_variances)/np.mean(block_variances):.3f}")
    
    print(f"\n  Bartlett's test for variance homogeneity:")
    print(f"  Statistic: {bartlett_stat:.4f}")
    print(f"  p-value: {bartlett_p:.6f}")
    print(f"  Verdict: {'⚠️  VARIANCE IS NOT CONSTANT (regime shifts?)' if bartlett_p < 0.05 else '✓ Variance is stable'}")
    
    # Levene's test (more robust)
    try:
        levene_stat, levene_p = stats.levene(*blocks)
    except Exception:
        levene_stat, levene_p = 0.0, 1.0
    print(f"\n  Levene's test (robust version):")
    print(f"  Statistic: {levene_stat:.4f}")
    print(f"  p-value: {levene_p:.6f}")
    print(f"  Verdict: {'⚠️  VARIANCE NOT CONSTANT' if levene_p < 0.05 else '✓ Variance is stable'}")
    
    return {'bartlett_p': bartlett_p, 'levene_p': levene_p, 
            'block_variances': block_variances}


# =====================================================================
# METHOD 16: RECURRENCE ANALYSIS
# =====================================================================

def method_16_recurrence(draws, max_number=49):
    """
    WHAT IT TESTS:
      Do identical or near-identical draws repeat more than expected?
    
    HOW IT WORKS:
      Check for exact draw repeats and near-matches (sharing 4+ numbers).
      Compare to Monte Carlo expected rate.
    
    WHAT RIGGING IT CATCHES:
      Draw recycling, limited randomness source, pattern repetition.
    """
    print("\n" + "=" * 70)
    print("METHOD 16: RECURRENCE / NEAR-MATCH ANALYSIS")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    
    # Convert draws to sets for fast comparison
    draw_sets = [frozenset(draw) for draw in draws]
    
    # Check for exact repeats
    draw_counter = Counter([tuple(sorted(d)) for d in draws])
    exact_repeats = {k: v for k, v in draw_counter.items() if v > 1}
    
    print(f"\n  Exact draw repeats: {len(exact_repeats)}")
    for draw, count in sorted(exact_repeats.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {draw}: appeared {count} times")
    
    # Near-matches (sharing N-1 numbers)
    match_threshold = n_per_draw - 1  # e.g., 4 out of 5
    near_matches = 0
    near_match_examples = []
    
    for i in range(min(n_draws, 500)):  # Limit for performance
        for j in range(i + 1, min(n_draws, 500)):
            overlap = len(draw_sets[i] & draw_sets[j])
            if overlap >= match_threshold:
                near_matches += 1
                if len(near_match_examples) < 5:
                    near_match_examples.append((i, j, overlap, 
                                                sorted(draw_sets[i]), 
                                                sorted(draw_sets[j])))
    
    print(f"\n  Near-matches (≥{match_threshold}/{n_per_draw} numbers shared):")
    print(f"  Found: {near_matches} (in first 500 draws)")
    for i, j, overlap, d1, d2 in near_match_examples:
        print(f"    Draw {i} {d1} & Draw {j} {d2}: {overlap} shared")
    
    # Monte Carlo comparison
    mc_near = []
    for _ in range(200):
        sim = [frozenset(np.random.choice(range(1, max_number + 1), 
               size=n_per_draw, replace=False)) for _ in range(min(500, n_draws))]
        nm = sum(1 for i in range(len(sim)) for j in range(i+1, len(sim)) 
                 if len(sim[i] & sim[j]) >= match_threshold)
        mc_near.append(nm)
    
    mc_mean = np.mean(mc_near)
    mc_std = np.std(mc_near)
    z = (near_matches - mc_mean) / mc_std if mc_std > 0 else 0
    
    print(f"\n  Monte Carlo: expected near-matches = {mc_mean:.1f} ± {mc_std:.1f}")
    print(f"  Z-score: {z:.2f}")
    print(f"  Verdict: {'⚠️  ABNORMAL RECURRENCE' if abs(z) > 2 else '✓ Normal recurrence rate'}")
    
    return {'exact_repeats': len(exact_repeats), 'near_matches': near_matches, 'z_score': z}


# =====================================================================
# METHOD 17: MACHINE LEARNING PREDICTABILITY TEST
# =====================================================================

def method_17_ml_predictability(draws, max_number=49):
    """
    WHAT IT TESTS:
      Can a machine learning model predict future draws better than chance?
    
    HOW IT WORKS:
      Train Random Forest / Gradient Boosting on features from previous
      draws to predict features of the next draw. If accuracy exceeds
      random baseline, there's a learnable pattern.
    
    WHAT RIGGING IT CATCHES:
      ANY systematic pattern, regardless of type. This is the
      "catch-all" method — if ML can predict it, something is wrong.
    """
    print("\n" + "=" * 70)
    print("METHOD 17: MACHINE LEARNING PREDICTABILITY TEST")
    print("=" * 70)
    
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    
    # Feature engineering: use last K draws to predict next
    lookback = 5
    
    # Create binary presence matrix
    presence = np.zeros((n_draws, max_number))
    for i, draw in enumerate(draws):
        for num in draw:
            presence[i, num - 1] = 1
    
    # Build features: flatten last K draws
    X = []
    y_sum = []  # Predict if sum is above/below median
    y_presence = []  # Predict if specific number appears
    
    target_number = 1  # Will test multiple
    
    for i in range(lookback, n_draws):
        features = presence[i - lookback:i].flatten()
        X.append(features)
        y_sum.append(draws[i].sum())
        y_presence.append(1 if target_number in draws[i] else 0)
    
    X = np.array(X)
    y_sum = np.array(y_sum)
    y_binary = (y_sum > np.median(y_sum)).astype(int)
    
    # Test 1: Predict above/below median sum
    print(f"\n  Test A: Predicting if draw sum is above/below median")
    print(f"  Features: presence matrix of last {lookback} draws ({X.shape[1]} features)")
    print(f"  Baseline (random): 50%")
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    scores = cross_val_score(rf, X, y_binary, cv=10, scoring='accuracy')
    print(f"  Random Forest: {scores.mean():.1%} ± {scores.std():.1%}")
    
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    scores_gb = cross_val_score(gb, X, y_binary, cv=10, scoring='accuracy')
    print(f"  Gradient Boosting: {scores_gb.mean():.1%} ± {scores_gb.std():.1%}")
    
    best_accuracy = max(scores.mean(), scores_gb.mean())
    
    # Test 2: Predict presence of specific numbers
    print(f"\n  Test B: Predicting individual number presence")
    print(f"  Baseline: {n_per_draw/max_number:.1%}")
    
    significant_numbers = []
    for num in range(1, min(max_number + 1, 50)):
        y_num = np.array([1 if num in draws[i] else 0 for i in range(lookback, n_draws)])
        base_rate = y_num.mean()
        
        if base_rate > 0.02 and base_rate < 0.98:  # Skip extreme imbalances
            try:
                rf_num = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
                scores_num = cross_val_score(rf_num, X, y_num, cv=5, scoring='accuracy')
                
                # Compare to baseline (always predict majority class)
                majority_baseline = max(base_rate, 1 - base_rate)
                improvement = scores_num.mean() - majority_baseline
                
                if improvement > 0.05:  # 5% improvement over baseline
                    significant_numbers.append((num, scores_num.mean(), majority_baseline, improvement))
                    print(f"    ⚠️  Number {num:2d}: accuracy={scores_num.mean():.1%}, "
                          f"baseline={majority_baseline:.1%}, improvement={improvement:+.1%}")
            except:
                continue
    
    print(f"\n  Numbers with predictable patterns: {len(significant_numbers)}/{max_number}")
    
    # Overall verdict
    if best_accuracy > 0.55 or len(significant_numbers) > 3:
        print(f"  Verdict: ⚠️  ML CAN PREDICT PATTERNS (accuracy: {best_accuracy:.1%})")
    else:
        print(f"  Verdict: ✓ No learnable patterns (accuracy: {best_accuracy:.1%})")
    
    return {'best_accuracy': best_accuracy, 'significant_numbers': significant_numbers}


# =====================================================================
# MASTER SUMMARY
# =====================================================================

def print_summary(results):
    """Print a comprehensive summary of all tests."""
    
    print("\n\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "  COMPREHENSIVE ANALYSIS SUMMARY  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│ METHOD                              │ RESULT                   │
├─────────────────────────────────────┼──────────────────────────┤""")
    
    tests = [
        ("1.  Frequency Analysis",        results.get('m01', {}).get('p_value', None), False),
        ("2.  Pair Co-occurrence",         results.get('m02', {}).get('p_value', None), False),
        ("3.  Triplet Repeats",            results.get('m03', {}).get('z_score', None), True),
        ("4.  Serial Autocorrelation",     len(results.get('m04', {}).get('significant_lags', [])), True),
        ("5.  Runs Test",                  results.get('m05', {}).get('p_value', None), False),
        ("6.  Gap Analysis",               results.get('m06', {}).get('ks_p', None), False),
        ("7.  Sum/Range Distribution",     results.get('m07', {}).get('var_ratio', None), True),
        ("8.  Spectral Analysis",          len(results.get('m08', {}).get('peak_indices', [])), True),
        ("9.  Conditional Probability",    results.get('m09', {}).get('n_significant', None), True),
        ("10. Mutual Information",         results.get('m10', {}).get('p_value', None), False),
        ("11. Benford's Law",              results.get('m11', {}).get('p_value', None), False),
        ("14. KS Distribution Test",       results.get('m14', {}).get('ks_p', None), False),
        ("15. Variance Ratio",             results.get('m15', {}).get('bartlett_p', None), False),
        ("16. Recurrence Analysis",        results.get('m16', {}).get('z_score', None), True),
        ("17. ML Predictability",          results.get('m17', {}).get('best_accuracy', None), True),
    ]
    
    suspicious_count = 0
    for name, value, is_higher_bad in tests:
        if value is None:
            status = "  --  "
        elif is_higher_bad:
            if isinstance(value, float):
                if name == "17. ML Predictability":
                    suspicious = value > 0.55
                elif name == "7.  Sum/Range Distribution":
                    suspicious = abs(value - 1.0) > 0.15
                else:
                    suspicious = abs(value) > 2
                status = f"⚠️  {value:.4f}" if suspicious else f"✓  {value:.4f}"
            else:
                suspicious = value > 3
                status = f"⚠️  {value}" if suspicious else f"✓  {value}"
        else:
            suspicious = value < 0.05
            status = f"⚠️  p={value:.4f}" if suspicious else f"✓  p={value:.4f}"
        
        if isinstance(value, (int, float)) and value is not None:
            if suspicious:
                suspicious_count += 1
        
        print(f"│ {name:<37s}│ {status:<24s} │")
    
    print(f"└─────────────────────────────────────┴──────────────────────────┘")
    
    print(f"\n  Tests showing anomalies: {suspicious_count}/{len(tests)}")
    
    if suspicious_count == 0:
        print(f"\n  🟢 OVERALL: No evidence of rigging detected by any method.")
    elif suspicious_count <= 2:
        print(f"\n  🟡 OVERALL: Weak signals detected. Could be noise or subtle manipulation.")
    elif suspicious_count <= 5:
        print(f"\n  🟠 OVERALL: Multiple anomalies detected. Further investigation warranted.")
    else:
        print(f"\n  🔴 OVERALL: Strong evidence of non-random patterns across multiple tests.")


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def run_full_analysis(filepath='Data.csv', sep=';', 
                       columns=['n1', 'n2', 'n3', 'n4', 'n5'],
                       max_number=49):
    """Run all 17 analysis methods."""
    
    print("╔" + "═" * 68 + "╗")
    print("║" + "  COMPREHENSIVE LOTTERY PATTERN DETECTION  ".center(68) + "║")
    print("║" + "  17 Methods × 1 Dataset  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Load data
    print("\n📊 LOADING DATA")
    draws = load_data(filepath, sep, columns)
    
    results = {}
    
    # Run all methods
    results['m01'] = method_01_frequency(draws, max_number)
    results['m02'] = method_02_pairs(draws, max_number)
    results['m03'] = method_03_triplets(draws, max_number)
    results['m04'] = method_04_autocorrelation(draws)
    results['m05'] = method_05_runs(draws)
    results['m06'] = method_06_gaps(draws, max_number)
    results['m07'] = method_07_sum_range(draws, max_number)
    results['m08'] = method_08_spectral(draws, max_number)
    results['m09'] = method_09_conditional(draws, max_number)
    results['m10'] = method_10_mutual_information(draws, max_number)
    results['m11'] = method_11_benford(draws)
    results['m12'] = method_12_balance(draws, max_number)
    results['m13'] = method_13_consecutive(draws, max_number)
    results['m14'] = method_14_ks_test(draws, max_number)
    results['m15'] = method_15_variance_ratio(draws)
    results['m16'] = method_16_recurrence(draws, max_number)
    results['m17'] = method_17_ml_predictability(draws, max_number)
    
    # Summary
    print_summary(results)
    
    return results


if __name__ == "__main__":
    results = run_full_analysis()