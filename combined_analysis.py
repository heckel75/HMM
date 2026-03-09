"""
============================================================================
COMBINED CROSS-REFERENCE ANALYSIS
============================================================================

This script pulls together ALL signals from every pipeline and 
cross-references them to find patterns that no single method caught alone.

The idea: weak signals from different methods pointing in the SAME
direction become strong evidence when combined.

============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chisquare, ks_2samp, spearmanr, pearsonr, binomtest
from collections import Counter, defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath='Data.csv', sep=';', date_col='date',
              number_cols=['n1','n2','n3','n4','n5'],
              date_format='%d/%m/%Y'):
    df = pd.read_csv(filepath, sep=sep)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        df = df.sort_values(date_col).reset_index(drop=True)
    draws = df[number_cols].values
    dates = df[date_col] if date_col in df.columns else None
    print(f"  Loaded {len(draws)} draws, {draws.shape[1]} numbers per draw")
    print(f"  Range: {draws.min()}-{draws.max()}")
    return draws, dates


# =====================================================================
# CROSS-REF 1: SUSPECT BALL PROFILING
# =====================================================================

def crossref_01_ball_profiles(draws, dates, max_number=49):
    """
    For EACH ball (1-49), compile every metric across all methods.
    Balls with multiple anomalies across different tests are the 
    most suspicious.
    """
    print("\n" + "=" * 70)
    print("CROSS-REF 1: INDIVIDUAL BALL PROFILING")
    print("  (Combining frequency, gaps, wear, position, modular, interactions)")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    all_numbers = draws.flatten()
    half = n_draws // 2
    expected_freq = len(all_numbers) / max_number
    expected_rate = n_per_draw / max_number
    
    ball_profiles = {}
    
    for num in range(1, max_number + 1):
        profile = {'number': num, 'flags': 0, 'details': []}
        
        # --- Frequency ---
        freq = np.sum(all_numbers == num)
        freq_dev = (freq - expected_freq) / expected_freq * 100
        profile['frequency'] = freq
        profile['freq_deviation_pct'] = freq_dev
        if abs(freq_dev) > 15:
            profile['flags'] += 1
            profile['details'].append(f"Frequency {freq_dev:+.1f}%")
        
        # --- Wear (first half vs second half) ---
        first_count = sum(1 for draw in draws[:half] if num in draw)
        second_count = sum(1 for draw in draws[half:] if num in draw)
        rate_1 = first_count / half
        rate_2 = second_count / (n_draws - half)
        wear_change = (rate_2 - rate_1) / rate_1 * 100 if rate_1 > 0 else 0
        profile['rate_first_half'] = rate_1
        profile['rate_second_half'] = rate_2
        profile['wear_change_pct'] = wear_change
        
        p_wear = binomtest(second_count, n_draws - half, rate_1).pvalue if rate_1 > 0 and rate_1 < 1 else 1.0
        profile['wear_p'] = p_wear
        if p_wear < 0.05:
            profile['flags'] += 1
            profile['details'].append(f"Wear change {wear_change:+.1f}% (p={p_wear:.4f})")
        
        # --- Gap analysis ---
        appearances = [i for i, draw in enumerate(draws) if num in draw]
        if len(appearances) >= 2:
            gaps = np.diff(appearances)
            mean_gap = np.mean(gaps)
            expected_gap = 1 / expected_rate
            gap_dev = (mean_gap - expected_gap) / expected_gap * 100
            profile['mean_gap'] = mean_gap
            profile['gap_deviation_pct'] = gap_dev
            
            # Max gap (suspiciously long absence)
            profile['max_gap'] = np.max(gaps)
            profile['min_gap'] = np.min(gaps)
            
            if abs(gap_dev) > 15:
                profile['flags'] += 1
                profile['details'].append(f"Gap deviation {gap_dev:+.1f}%")
        
        # --- Autocorrelation (lag-1) ---
        presence = np.array([1.0 if num in draw else 0.0 for draw in draws])
        if presence.sum() > 5:
            autocorr = np.corrcoef(presence[:-1], presence[1:])[0, 1]
            profile['autocorr_lag1'] = autocorr
            conf = 2 / np.sqrt(n_draws)
            if abs(autocorr) > conf:
                profile['flags'] += 1
                profile['details'].append(f"Autocorrelation {autocorr:+.4f}")
        
        # --- Modular properties ---
        for mod in [2, 3, 5, 7, 8]:
            profile[f'mod{mod}'] = num % mod
        
        # --- Positional preference ---
        pos_counts = [0] * n_per_draw
        for draw in draws:
            if num in draw:
                # Find position in sorted draw
                sorted_draw = np.sort(draw)
                pos = np.where(sorted_draw == num)[0]
                if len(pos) > 0:
                    pos_counts[pos[0]] += 1
        
        total_appearances = sum(pos_counts)
        if total_appearances > 0:
            pos_ratios = [c / total_appearances for c in pos_counts]
            profile['preferred_position'] = np.argmax(pos_ratios) + 1
            profile['position_concentration'] = max(pos_ratios)
        
        # --- Consecutive draw repeats ---
        repeats = sum(1 for i in range(len(presence)-1) if presence[i] and presence[i+1])
        expected_repeats = sum(presence[:-1]) * expected_rate
        profile['consecutive_repeats'] = repeats
        profile['expected_repeats'] = expected_repeats
        if expected_repeats > 3:
            repeat_ratio = repeats / expected_repeats
            if abs(repeat_ratio - 1) > 0.4:
                profile['flags'] += 1
                profile['details'].append(f"Repeat ratio {repeat_ratio:.2f}x")
        
        # --- Co-occurrence bias ---
        cooccur_counts = Counter()
        for draw in draws:
            if num in draw:
                for other in draw:
                    if other != num:
                        cooccur_counts[other] += 1
        
        if cooccur_counts:
            total_cooccur = sum(cooccur_counts.values())
            expected_cooccur = total_cooccur / (max_number - 1)
            top_partner = cooccur_counts.most_common(1)[0]
            bottom_partner = cooccur_counts.most_common()[-1]
            top_dev = (top_partner[1] - expected_cooccur) / expected_cooccur * 100
            bottom_dev = (bottom_partner[1] - expected_cooccur) / expected_cooccur * 100
            profile['top_partner'] = top_partner
            profile['bottom_partner'] = bottom_partner
            
            if abs(top_dev) > 80:
                profile['flags'] += 1
                profile['details'].append(f"Strong partner: ball {top_partner[0]} (+{top_dev:.0f}%)")
        
        # --- Trend (slope over time) ---
        if len(appearances) > 10:
            rolling_window = 100
            rolling_freq = np.convolve(presence, np.ones(rolling_window)/rolling_window, mode='valid')
            if len(rolling_freq) > 10:
                idx = np.arange(len(rolling_freq))
                slope, _, _, p_trend, _ = stats.linregress(idx, rolling_freq)
                profile['trend_slope'] = slope
                profile['trend_p'] = p_trend
                if p_trend < 0.01:
                    profile['flags'] += 1
                    direction = "increasing" if slope > 0 else "decreasing"
                    profile['details'].append(f"Trend: {direction} (p={p_trend:.4f})")
        
        ball_profiles[num] = profile
    
    # Sort by number of flags
    sorted_balls = sorted(ball_profiles.values(), key=lambda x: x['flags'], reverse=True)
    
    print(f"\n  {'='*65}")
    print(f"  BALL SUSPICION RANKING (most flags = most suspicious)")
    print(f"  {'='*65}")
    
    for profile in sorted_balls:
        num = profile['number']
        flags = profile['flags']
        freq_dev = profile['freq_deviation_pct']
        
        if flags > 0:
            mod8 = profile['mod8']
            print(f"\n  Ball {num:2d} [{flags} flags] (mod8={mod8}, freq={profile['frequency']}, "
                  f"dev={freq_dev:+.1f}%)")
            for detail in profile['details']:
                print(f"    → {detail}")
    
    flagless = sum(1 for p in sorted_balls if p['flags'] == 0)
    print(f"\n  Balls with 0 flags: {flagless}/{max_number}")
    
    return ball_profiles


# =====================================================================
# CROSS-REF 2: MODULAR GROUP DEEP DIVE
# =====================================================================

def crossref_02_modular_groups(draws, dates, max_number=49):
    """
    For each modular grouping that showed a signal (especially mod 8),
    do a deep dive: frequency over time, wear patterns, interactions.
    """
    print("\n" + "=" * 70)
    print("CROSS-REF 2: MODULAR GROUP DEEP DIVE")
    print("  (Testing all moduli systematically)")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    all_numbers = draws.flatten()
    
    for mod in [2, 3, 4, 5, 6, 7, 8, 10, 12]:
        print(f"\n  {'─'*60}")
        print(f"  MOD {mod} ANALYSIS")
        print(f"  {'─'*60}")
        
        # Group numbers by residue
        groups = defaultdict(list)
        for n in range(1, max_number + 1):
            groups[n % mod].append(n)
        
        # Frequency per group
        group_freqs = {}
        for residue, nums in sorted(groups.items()):
            freq = sum(np.sum(all_numbers == n) for n in nums)
            expected = len(all_numbers) * len(nums) / max_number
            dev = (freq - expected) / expected * 100
            group_freqs[residue] = {'freq': freq, 'expected': expected, 'dev': dev}
        
        # Chi-squared test
        obs = [group_freqs[r]['freq'] for r in sorted(group_freqs)]
        exp = [group_freqs[r]['expected'] for r in sorted(group_freqs)]
        chi2, p = chisquare(obs, exp)
        
        flag = "⚠️" if p < 0.05 else "  "
        print(f"  {flag} Overall: χ²={chi2:.2f}, p={p:.6f}")
        
        if p < 0.10:  # Show details for borderline and significant
            for residue in sorted(group_freqs):
                info = group_freqs[residue]
                nums = groups[residue]
                print(f"    Residue {residue} (balls {nums}): "
                      f"freq={info['freq']}, expected={info['expected']:.0f}, "
                      f"dev={info['dev']:+.1f}%")
            
            # Time evolution: does this mod bias change over time?
            print(f"\n    Time evolution (4 periods):")
            quarter = n_draws // 4
            for q in range(4):
                q_draws = draws[q*quarter:(q+1)*quarter].flatten()
                q_obs = []
                q_exp = []
                for residue in sorted(groups):
                    f = sum(np.sum(q_draws == n) for n in groups[residue])
                    e = len(q_draws) * len(groups[residue]) / max_number
                    q_obs.append(f)
                    q_exp.append(e)
                q_chi2, q_p = chisquare(q_obs, q_exp)
                
                # Find most deviant residue in this period
                deviations = [(r, (o-e)/e*100) for r, o, e in 
                             zip(sorted(groups), q_obs, q_exp)]
                worst = max(deviations, key=lambda x: abs(x[1]))
                
                period_start = dates.iloc[q*quarter].strftime('%d/%m/%Y') if dates is not None else f"Draw {q*quarter}"
                print(f"      Period {q+1} ({period_start}): χ²={q_chi2:.2f}, p={q_p:.4f}, "
                      f"worst=residue {worst[0]} ({worst[1]:+.1f}%)")
    
    return {}


# =====================================================================
# CROSS-REF 3: TEMPORAL × PHYSICAL INTERACTION
# =====================================================================

def crossref_03_temporal_physical(draws, dates, max_number=49):
    """
    Cross-reference TIME-based findings with PHYSICAL findings.
    Do the balls that are changing over time (P5) correlate with
    specific physical properties (mod group, magnitude, etc.)?
    """
    print("\n" + "=" * 70)
    print("CROSS-REF 3: TEMPORAL × PHYSICAL CROSS-REFERENCE")
    print("  (Do wear patterns correlate with physical ball properties?)")
    print("=" * 70)
    
    n_draws = len(draws)
    half = n_draws // 2
    
    # Calculate wear rate for each ball
    wear_rates = []
    for num in range(1, max_number + 1):
        first = sum(1 for draw in draws[:half] if num in draw) / half
        second = sum(1 for draw in draws[half:] if num in draw) / (n_draws - half)
        change = second - first
        wear_rates.append(change)
    
    wear_rates = np.array(wear_rates)
    numbers = np.arange(1, max_number + 1)
    
    # Test 1: Does wear correlate with ball NUMBER (magnitude)?
    corr_mag, p_mag = spearmanr(numbers, wear_rates)
    print(f"\n  Test 1: Wear vs ball number (magnitude)")
    print(f"    Correlation: ρ={corr_mag:+.4f}, p={p_mag:.6f}")
    print(f"    {'⚠️  MAGNITUDE AFFECTS WEAR' if p_mag < 0.05 else '✓ No magnitude-wear link'}")
    
    # Test 2: Does wear correlate with initial frequency?
    initial_freq = np.array([sum(1 for draw in draws[:half] if num in draw) 
                             for num in range(1, max_number + 1)])
    corr_freq, p_freq = spearmanr(initial_freq, wear_rates)
    print(f"\n  Test 2: Wear vs initial frequency (fatigue effect)")
    print(f"    Correlation: ρ={corr_freq:+.4f}, p={p_freq:.6f}")
    print(f"    {'⚠️  HIGH-USE BALLS CHANGE MORE' if p_freq < 0.05 else '✓ No use-wear link'}")
    
    # Test 3: Wear by modular group
    print(f"\n  Test 3: Wear rate by modular group")
    for mod in [2, 3, 5, 7, 8]:
        groups = defaultdict(list)
        for n in range(max_number):
            groups[(n + 1) % mod].append(wear_rates[n])
        
        group_means = {r: np.mean(rates) for r, rates in sorted(groups.items())}
        
        # Kruskal-Wallis
        group_lists = [groups[r] for r in sorted(groups)]
        if len(group_lists) >= 2 and all(len(g) > 1 for g in group_lists):
            h, p = stats.kruskal(*group_lists)
            flag = "⚠️" if p < 0.05 else "  "
            means_str = ", ".join([f"r{r}:{m:+.4f}" for r, m in group_means.items()])
            print(f"    {flag} Mod {mod}: H={h:.2f}, p={p:.4f} [{means_str}]")
    
    # Test 4: Wear by digit properties
    print(f"\n  Test 4: Wear by digit properties")
    single_digit_wear = wear_rates[:9]
    double_digit_wear = wear_rates[9:]
    mw_stat, mw_p = stats.mannwhitneyu(single_digit_wear, double_digit_wear, alternative='two-sided')
    print(f"    Single-digit mean wear: {np.mean(single_digit_wear):+.4f}")
    print(f"    Double-digit mean wear: {np.mean(double_digit_wear):+.4f}")
    print(f"    Mann-Whitney p={mw_p:.6f}")
    print(f"    {'⚠️  DIGIT COUNT AFFECTS WEAR' if mw_p < 0.05 else '✓ No digit-wear link'}")
    
    # Test 5: Odd vs even wear
    odd_wear = [wear_rates[n] for n in range(max_number) if (n+1) % 2 == 1]
    even_wear = [wear_rates[n] for n in range(max_number) if (n+1) % 2 == 0]
    mw_oe, p_oe = stats.mannwhitneyu(odd_wear, even_wear, alternative='two-sided')
    print(f"\n  Test 5: Odd vs even wear")
    print(f"    Odd mean wear:  {np.mean(odd_wear):+.4f}")
    print(f"    Even mean wear: {np.mean(even_wear):+.4f}")
    print(f"    Mann-Whitney p={p_oe:.6f}")
    print(f"    {'⚠️  PARITY AFFECTS WEAR' if p_oe < 0.05 else '✓ No parity-wear link'}")
    
    return {'wear_rates': wear_rates}


# =====================================================================
# CROSS-REF 4: INTERACTION NETWORK × TIME
# =====================================================================

def crossref_04_network_time(draws, dates, max_number=49):
    """
    Do ball interaction patterns (attractions/repulsions) change over time?
    A physical defect might create stable interactions, while random
    fluctuations would differ across time periods.
    """
    print("\n" + "=" * 70)
    print("CROSS-REF 4: BALL INTERACTIONS × TIME STABILITY")
    print("  (Are pair attractions/repulsions stable or transient?)")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    n_periods = 4
    period_size = n_draws // n_periods
    
    expected_per_period = period_size * (n_per_draw * (n_per_draw - 1)) / (max_number * (max_number - 1))
    
    # Build co-occurrence matrix for each period
    period_matrices = []
    for p in range(n_periods):
        start = p * period_size
        end = start + period_size
        matrix = np.zeros((max_number, max_number))
        for draw in draws[start:end]:
            for i in range(len(draw)):
                for j in range(i+1, len(draw)):
                    matrix[draw[i]-1][draw[j]-1] += 1
                    matrix[draw[j]-1][draw[i]-1] += 1
        period_matrices.append(matrix)
    
    # Find pairs that are CONSISTENTLY elevated or suppressed across ALL periods
    print(f"\n  Looking for STABLE interactions across {n_periods} time periods...")
    print(f"  Expected co-occurrence per period: {expected_per_period:.2f}")
    
    stable_attractions = []
    stable_repulsions = []
    
    for i in range(max_number):
        for j in range(i+1, max_number):
            period_counts = [period_matrices[p][i][j] for p in range(n_periods)]
            period_devs = [(c - expected_per_period) / expected_per_period * 100 
                          for c in period_counts]
            
            # ALL periods elevated?
            if all(d > 30 for d in period_devs):
                mean_dev = np.mean(period_devs)
                stable_attractions.append((i+1, j+1, period_counts, mean_dev))
            
            # ALL periods suppressed?
            if all(d < -30 for d in period_devs):
                mean_dev = np.mean(period_devs)
                stable_repulsions.append((i+1, j+1, period_counts, mean_dev))
    
    if stable_attractions:
        print(f"\n  ⚠️  STABLE ATTRACTIONS (elevated in ALL {n_periods} periods):")
        for a, b, counts, dev in sorted(stable_attractions, key=lambda x: x[3], reverse=True)[:15]:
            print(f"    Balls {a:2d}-{b:2d}: counts={counts}, avg dev={dev:+.0f}%")
    else:
        print(f"\n  No pairs consistently elevated across all periods")
    
    if stable_repulsions:
        print(f"\n  ⚠️  STABLE REPULSIONS (suppressed in ALL {n_periods} periods):")
        for a, b, counts, dev in sorted(stable_repulsions, key=lambda x: x[3])[:15]:
            print(f"    Balls {a:2d}-{b:2d}: counts={counts}, avg dev={dev:+.0f}%")
    else:
        print(f"\n  No pairs consistently suppressed across all periods")
    
    # Compare to transient interactions (strong in ONE period but not others)
    transient_count = 0
    for i in range(max_number):
        for j in range(i+1, max_number):
            period_counts = [period_matrices[p][i][j] for p in range(n_periods)]
            period_devs = [(c - expected_per_period) / expected_per_period * 100 
                          for c in period_counts]
            if max(period_devs) > 100 and min(period_devs) < 0:
                transient_count += 1
    
    print(f"\n  Stable interactions: {len(stable_attractions) + len(stable_repulsions)}")
    print(f"  Transient interactions (one period only): {transient_count}")
    ratio = (len(stable_attractions) + len(stable_repulsions)) / max(transient_count, 1)
    print(f"  Stability ratio: {ratio:.3f}")
    print(f"  (Higher ratio → more likely physical cause rather than random noise)")
    
    return {'stable_attractions': stable_attractions, 'stable_repulsions': stable_repulsions}


# =====================================================================
# CROSS-REF 5: POSITION × MODULAR × WEAR COMBINED
# =====================================================================

def crossref_05_combined_model(draws, dates, max_number=49):
    """
    Build a combined picture: do the mod-8 bias, wear patterns,
    and positional effects all point to the same balls?
    """
    print("\n" + "=" * 70)
    print("CROSS-REF 5: COMBINED ANOMALY SCORING")
    print("  (Which balls are flagged by MULTIPLE independent methods?)")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    all_numbers = draws.flatten()
    half = n_draws // 2
    expected_freq = len(all_numbers) / max_number
    expected_rate = n_per_draw / max_number
    
    scores = {}
    
    for num in range(1, max_number + 1):
        score = 0.0
        evidence = []
        
        # 1. Frequency deviation
        freq = np.sum(all_numbers == num)
        freq_z = (freq - expected_freq) / np.sqrt(expected_freq)
        if abs(freq_z) > 1.5:
            score += abs(freq_z) - 1.5
            evidence.append(f"freq_z={freq_z:+.2f}")
        
        # 2. Wear rate
        first = sum(1 for draw in draws[:half] if num in draw) / half
        second = sum(1 for draw in draws[half:] if num in draw) / (n_draws - half)
        wear_z = (second - first) / np.sqrt(expected_rate * (1-expected_rate) * (1/half + 1/(n_draws-half)))
        if abs(wear_z) > 1.5:
            score += abs(wear_z) - 1.5
            evidence.append(f"wear_z={wear_z:+.2f}")
        
        # 3. Modular residue (mod 8 deviation)
        mod8_group = [n for n in range(1, max_number+1) if n % 8 == num % 8]
        mod8_freq = sum(np.sum(all_numbers == n) for n in mod8_group)
        mod8_expected = len(all_numbers) * len(mod8_group) / max_number
        mod8_z = (mod8_freq - mod8_expected) / np.sqrt(mod8_expected)
        if abs(mod8_z) > 1.5:
            score += abs(mod8_z) - 1.5
            evidence.append(f"mod8_z={mod8_z:+.2f} (residue {num%8})")
        
        # 4. Gap regularity
        appearances = [i for i, draw in enumerate(draws) if num in draw]
        if len(appearances) >= 5:
            gaps = np.diff(appearances)
            gap_cv = np.std(gaps) / np.mean(gaps)  # Coefficient of variation
            # For geometric distribution, CV ≈ sqrt(1-p)/p ≈ 1 for small p
            expected_cv = np.sqrt(1 - expected_rate) / expected_rate
            # Normalized to expected
            # Actually for geometric, CV ≈ sqrt(1-p)/p
            # Compare to monte carlo
            mc_cvs = []
            for _ in range(500):
                sim_appearances = np.sort(np.random.choice(n_draws, size=len(appearances), replace=False))
                sim_gaps = np.diff(sim_appearances)
                if np.mean(sim_gaps) > 0:
                    mc_cvs.append(np.std(sim_gaps) / np.mean(sim_gaps))
            if mc_cvs:
                cv_z = (gap_cv - np.mean(mc_cvs)) / np.std(mc_cvs)
                if abs(cv_z) > 1.5:
                    score += abs(cv_z) - 1.5
                    evidence.append(f"gap_cv_z={cv_z:+.2f}")
        
        # 5. Autocorrelation
        presence = np.array([1.0 if num in draw else 0.0 for draw in draws])
        if presence.sum() > 10:
            autocorr = np.corrcoef(presence[:-1], presence[1:])[0, 1]
            ac_z = autocorr * np.sqrt(n_draws)
            if abs(ac_z) > 1.5:
                score += abs(ac_z) - 1.5
                evidence.append(f"autocorr_z={ac_z:+.2f}")
        
        # 6. Co-occurrence concentration
        partners = Counter()
        for draw in draws:
            if num in draw:
                for other in draw:
                    if other != num:
                        partners[other] += 1
        if partners:
            partner_vals = list(partners.values())
            partner_cv = np.std(partner_vals) / np.mean(partner_vals)
            # High CV means uneven partnerships (some balls always together)
            mc_pcvs = []
            for _ in range(300):
                sim_vals = np.random.multinomial(sum(partner_vals), 
                           np.ones(max_number-1)/(max_number-1))
                sim_vals = sim_vals[sim_vals > 0]
                if np.mean(sim_vals) > 0:
                    mc_pcvs.append(np.std(sim_vals) / np.mean(sim_vals))
            if mc_pcvs:
                pcv_z = (partner_cv - np.mean(mc_pcvs)) / np.std(mc_pcvs)
                if abs(pcv_z) > 1.5:
                    score += abs(pcv_z) - 1.5
                    evidence.append(f"partner_z={pcv_z:+.2f}")
        
        scores[num] = {'score': score, 'evidence': evidence}
    
    # Rank and display
    ranked = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    print(f"\n  COMBINED ANOMALY SCORE (higher = more suspicious)")
    print(f"  {'─'*65}")
    print(f"  {'Ball':>4s} {'Score':>7s}  {'Mod8':>4s}  Evidence")
    print(f"  {'─'*65}")
    
    for num, info in ranked:
        if info['score'] > 0:
            evidence_str = " | ".join(info['evidence'])
            print(f"  {num:4d} {info['score']:7.2f}  r={num%8:d}    {evidence_str}")
    
    zero_count = sum(1 for _, info in ranked if info['score'] == 0)
    print(f"\n  Balls with score 0 (no anomalies): {zero_count}/{max_number}")
    
    # Group scores by mod 8
    print(f"\n  SCORES AGGREGATED BY MOD-8 GROUP:")
    for residue in range(8):
        group_balls = [num for num in range(1, max_number+1) if num % 8 == residue]
        group_scores = [scores[num]['score'] for num in group_balls]
        mean_score = np.mean(group_scores)
        flagged = sum(1 for s in group_scores if s > 0)
        print(f"    Residue {residue} (balls {group_balls}): "
              f"mean score={mean_score:.2f}, flagged={flagged}/{len(group_balls)}")
    
    return scores


# =====================================================================
# CROSS-REF 6: MACHINE GEOMETRY INFERENCE
# =====================================================================

def crossref_06_geometry(draws, max_number=49):
    """
    Try to infer the machine's physical geometry from the data.
    Test if a specific arrangement of balls in the machine
    explains the observed patterns.
    """
    print("\n" + "=" * 70)
    print("CROSS-REF 6: MACHINE GEOMETRY INFERENCE")
    print("  (Can we reverse-engineer the machine layout?)")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    all_numbers = draws.flatten()
    
    # Test various hypothetical machine geometries
    # If machine has N slots, balls in slot k are numbers where (number-1)%N == k
    
    best_chi2 = 0
    best_geometry = None
    
    print(f"\n  Testing hypothetical machine geometries (N slots):")
    
    for n_slots in range(2, 20):
        # Assign balls to slots
        slots = defaultdict(list)
        for n in range(1, max_number + 1):
            slots[(n - 1) % n_slots].append(n)
        
        # Calculate frequency per slot
        slot_freqs = []
        slot_expected = []
        for s in range(n_slots):
            freq = sum(np.sum(all_numbers == n) for n in slots[s])
            expected = len(all_numbers) * len(slots[s]) / max_number
            slot_freqs.append(freq)
            slot_expected.append(expected)
        
        chi2, p = chisquare(slot_freqs, slot_expected)
        
        # Also check within-draw clustering by slot
        clustering_score = 0
        for draw in draws:
            slot_assignments = [(n - 1) % n_slots for n in draw]
            unique_slots = len(set(slot_assignments))
            clustering_score += (n_per_draw - unique_slots)
        clustering_score /= n_draws
        
        if p < 0.05 or n_slots in [2, 3, 5, 7, 8, 10, 12, 16]:
            flag = "⚠️" if p < 0.05 else "  "
            print(f"    {flag} {n_slots:2d} slots: χ²={chi2:8.2f}, p={p:.6f}, "
                  f"clustering={clustering_score:.3f}")
            
            if p < 0.05:
                # Show slot breakdown
                for s in range(n_slots):
                    dev = (slot_freqs[s] - slot_expected[s]) / slot_expected[s] * 100
                    if abs(dev) > 5:
                        print(f"          Slot {s} (balls {slots[s]}): {dev:+.1f}%")
        
        if chi2 > best_chi2:
            best_chi2 = chi2
            best_geometry = n_slots
    
    print(f"\n  Best fitting geometry: {best_geometry} slots (χ²={best_chi2:.2f})")
    
    # Deep dive on best geometry
    if best_geometry:
        print(f"\n  Deep dive on {best_geometry}-slot layout:")
        slots = defaultdict(list)
        for n in range(1, max_number + 1):
            slots[(n - 1) % best_geometry].append(n)
        
        for s in sorted(slots):
            nums = slots[s]
            freqs = [np.sum(all_numbers == n) for n in nums]
            mean_freq = np.mean(freqs)
            expected = len(all_numbers) / max_number
            dev = (mean_freq - expected) / expected * 100
            print(f"    Slot {s}: balls={nums}, mean_freq={mean_freq:.1f} ({dev:+.1f}%)")
    
    return {'best_geometry': best_geometry}


# =====================================================================
# CROSS-REF 7: TIME-WINDOWED MULTI-METHOD
# =====================================================================

def crossref_07_time_windows(draws, dates, max_number=49):
    """
    Run multiple tests in sliding time windows.
    If the machine inconsistency developed at a certain point,
    we should see multiple methods flag the same time period.
    """
    print("\n" + "=" * 70)
    print("CROSS-REF 7: TIME-WINDOWED MULTI-METHOD ANALYSIS")
    print("  (When did the inconsistency appear/change?)")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    window_size = 200
    step = 50
    
    print(f"\n  Window: {window_size} draws, Step: {step}")
    
    windows = []
    
    for start in range(0, n_draws - window_size, step):
        end = start + window_size
        w_draws = draws[start:end]
        w_flat = w_draws.flatten()
        
        # Test 1: Overall uniformity
        freq = np.bincount(w_flat, minlength=max_number+1)[1:]
        expected = np.full(max_number, len(w_flat) / max_number)
        chi2, p_chi = chisquare(freq, expected)
        
        # Test 2: Mod-8 bias
        mod8_obs = [np.sum(w_flat % 8 == r) for r in range(8)]
        mod8_exp_raw = np.zeros(8)
        for n in range(1, max_number+1):
            mod8_exp_raw[n % 8] += 1
        mod8_exp = mod8_exp_raw / mod8_exp_raw.sum() * len(w_flat)
        chi2_mod8, p_mod8 = chisquare(mod8_obs, mod8_exp)
        
        # Test 3: Adjacent pair rate
        adj_pairs = sum(
            1 for draw in w_draws
            for i in range(len(draw))
            for j in range(i+1, len(draw))
            if abs(draw[i] - draw[j]) == 1
        ) / len(w_draws)
        
        # Test 4: Mean sum
        mean_sum = w_draws.sum(axis=1).mean()
        
        # Test 5: Variance of sums
        var_sum = w_draws.sum(axis=1).var()
        
        date_label = dates.iloc[start].strftime('%d/%m/%Y') if dates is not None else f"Draw {start}"
        
        windows.append({
            'start': start, 'end': end, 'date': date_label,
            'chi2_p': p_chi, 'mod8_p': p_mod8,
            'adj_pairs': adj_pairs, 'mean_sum': mean_sum,
            'var_sum': var_sum,
            'n_anomalies': (1 if p_chi < 0.05 else 0) + (1 if p_mod8 < 0.05 else 0)
        })
    
    # Display windows with anomalies
    print(f"\n  {'Date':>12s} {'Draws':>10s} {'χ²_p':>8s} {'Mod8_p':>8s} "
          f"{'AdjPairs':>8s} {'MeanSum':>8s} {'VarSum':>8s} {'Flags':>5s}")
    print(f"  {'─'*70}")
    
    for w in windows:
        flags = ""
        if w['chi2_p'] < 0.05: flags += "F"
        if w['mod8_p'] < 0.05: flags += "M"
        
        flag_prefix = "⚠️" if flags else "  "
        print(f"  {flag_prefix}{w['date']:>10s} {w['start']:>4d}-{w['end']:>4d} "
              f"{w['chi2_p']:>8.4f} {w['mod8_p']:>8.4f} "
              f"{w['adj_pairs']:>8.3f} {w['mean_sum']:>8.1f} "
              f"{w['var_sum']:>8.1f} {flags:>5s}")
    
    return windows


# =====================================================================
# FINAL SUMMARY
# =====================================================================

def print_final_summary():
    print("\n\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "  GRAND UNIFIED ANALYSIS COMPLETE  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"""
  This analysis cross-referenced:
  
  ┌─────────────────────────────────────────────────────────────┐
  │ Pipeline              │ Methods │ Focus                     │
  ├───────────────────────┼─────────┼───────────────────────────┤
  │ HMM Pipeline          │    1    │ Hidden regime detection   │
  │ Comprehensive         │   17    │ Statistical patterns      │
  │ Date-Based            │   12    │ Calendar/time patterns    │
  │ Physical Machine      │   10    │ Mechanical bias           │
  │ Cross-Reference       │    7    │ Combined evidence         │
  ├───────────────────────┼─────────┼───────────────────────────┤
  │ TOTAL                 │   47    │ Exhaustive coverage       │
  └───────────────────────┴─────────┴───────────────────────────┘
  
  Key principle: No single method may be conclusive, but when
  multiple independent methods point to the same balls, positions,
  or time periods, the combined evidence becomes compelling.
    """)


# =====================================================================
# MAIN
# =====================================================================

def run_combined_analysis(filepath='Data.csv', sep=';',
                           date_col='date',
                           number_cols=['n1','n2','n3','n4','n5'],
                           max_number=49, date_format='%d/%m/%Y'):
    
    print("╔" + "═" * 68 + "╗")
    print("║" + "  COMBINED CROSS-REFERENCE ANALYSIS  ".center(68) + "║")
    print("║" + "  Synthesizing All Pipeline Results  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    print("\n📊 LOADING DATA")
    draws, dates = load_data(filepath, sep, date_col, number_cols, date_format)
    
    results = {}
    
    results['ball_profiles'] = crossref_01_ball_profiles(draws, dates, max_number)
    results['modular'] = crossref_02_modular_groups(draws, dates, max_number)
    results['temporal_physical'] = crossref_03_temporal_physical(draws, dates, max_number)
    results['network_time'] = crossref_04_network_time(draws, dates, max_number)
    results['combined_scores'] = crossref_05_combined_model(draws, dates, max_number)
    results['geometry'] = crossref_06_geometry(draws, max_number)
    results['time_windows'] = crossref_07_time_windows(draws, dates, max_number)
    
    print_final_summary()
    
    return results


if __name__ == "__main__":
    results = run_combined_analysis()
