"""
============================================================================
PHYSICAL MACHINE BIAS DETECTION
============================================================================

These methods test for biases caused by PHYSICAL properties of the machine
and balls, not abstract number patterns.

Physical machine inconsistencies include:
  - Ball weight/size variation (heavier balls sink, lighter rise)
  - Ball surface wear (older balls slide differently)
  - Manufacturing batch effects (balls made together behave similarly)
  - Paint/ink weight (double-digit numbers have more ink = heavier)
  - Machine pocket/slot bias (certain exit positions favored)
  - Mixing asymmetry (balls in certain positions mix poorly)
  - Draw order effects (first ball out vs last ball out)
  - Ball interaction (certain balls together change dynamics)
  - Temperature/humidity effects over time

============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chisquare, ks_2samp, spearmanr, pearsonr
from collections import Counter, defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath='Data.csv', sep=';', date_col='date',
              number_cols=None, date_format='%d/%m/%Y'):
    df = pd.read_csv(filepath, sep=sep)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        df = df.sort_values(date_col).reset_index(drop=True)
    if number_cols is None:
        number_cols = [c for c in df.columns if c != date_col]
    draws = df[number_cols].values
    dates = df[date_col] if date_col in df.columns else None
    print(f"  Loaded {len(draws)} draws, {draws.shape[1]} numbers per draw")
    return draws, dates, df


# =====================================================================
# P1: POSITIONAL BIAS (Draw Order)
# =====================================================================

def method_p01_positional(draws, max_number=49):
    """
    WHAT IT TESTS:
      Does each POSITION (1st drawn, 2nd drawn, etc.) favor certain numbers?
      
    WHY IT MATTERS FOR PHYSICAL MACHINES:
      If the machine has a mechanical bias, it might affect WHICH ball
      comes out first vs last. The first ball exits under different
      conditions than the fifth (fewer balls remaining, different pressure).
      
      If data is sorted (not draw order), this tests if certain numbers
      consistently appear as the smallest/largest in a draw, which can
      also reveal bias.
    """
    print("\n" + "=" * 70)
    print("METHOD P1: POSITIONAL BIAS ANALYSIS")
    print("=" * 70)
    
    n_positions = draws.shape[1]
    
    print(f"\n  Note: Testing if columns n1-n{n_positions} have different distributions.")
    print(f"  If data is SORTED: tests if certain numbers dominate low/high positions.")
    print(f"  If data is DRAW ORDER: tests if certain balls exit first/last.\n")
    
    # Per-position statistics
    for pos in range(n_positions):
        col = draws[:, pos]
        print(f"  Position {pos+1}: mean={np.mean(col):.1f}, median={np.median(col):.0f}, "
              f"std={np.std(col):.1f}, range=[{col.min()}-{col.max()}]")
    
    # Per-position frequency distribution
    print(f"\n  Per-position top 5 most frequent numbers:")
    position_biases = {}
    for pos in range(n_positions):
        col = draws[:, pos]
        freq = Counter(col)
        top5 = freq.most_common(5)
        expected = len(draws) / max_number
        
        # Chi-squared for this position
        obs = np.bincount(col, minlength=max_number + 1)[1:]
        # Expected depends on position if sorted
        chi2, p = chisquare(obs[obs > 0])  # Simple uniformity test on observed values
        
        position_biases[pos] = {'top5': top5, 'chi2': chi2, 'p': p}
        nums_str = ", ".join([f"{n}({c})" for n, c in top5])
        print(f"    Pos {pos+1}: {nums_str}  (χ²={chi2:.1f}, p={p:.4f})")
    
    # Cross-position correlation
    print(f"\n  Cross-position correlations:")
    for i in range(n_positions):
        for j in range(i+1, n_positions):
            corr, p = spearmanr(draws[:, i], draws[:, j])
            flag = "⚠️" if p < 0.01 else "  "
            print(f"    {flag} Pos {i+1} vs Pos {j+1}: ρ={corr:+.4f}, p={p:.6f}")
    
    return position_biases


# =====================================================================
# P2: MANUFACTURING BATCH EFFECTS
# =====================================================================

def method_p02_batch(draws, max_number=49):
    """
    WHAT IT TESTS:
      Do numbers from the same "batch" appear together more often?
      
    WHY IT MATTERS FOR PHYSICAL MACHINES:
      Balls are manufactured in batches (e.g., 1-10, 11-20, etc.).
      Balls from the same batch may have similar weight, size, or
      surface properties, causing them to behave similarly in the machine.
      
    BATCHES TESTED:
      - Decades: 1-9, 10-19, 20-29, 30-39, 40-49
      - Halves: 1-24, 25-49
      - Thirds: 1-16, 17-33, 34-49
      - Quintiles: 1-10, 11-20, 21-30, 31-40, 41-49
    """
    print("\n" + "=" * 70)
    print("METHOD P2: MANUFACTURING BATCH EFFECTS")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    
    # Define batch groupings
    batch_schemes = {
        'Decades (1-9,10-19,...)': lambda x: (x - 1) // 10,
        'Quintiles (1-10,11-20,...)': lambda x: (x - 1) // 10,
        'Thirds (1-16,17-33,34-49)': lambda x: 0 if x <= 16 else (1 if x <= 33 else 2),
        'Units digit (same last digit)': lambda x: x % 10,
        'Tens digit': lambda x: x // 10,
    }
    
    for scheme_name, batch_fn in batch_schemes.items():
        print(f"\n  Batch scheme: {scheme_name}")
        
        # Count how many balls from same batch appear together
        same_batch_counts = []
        for draw in draws:
            batches = [batch_fn(n) for n in draw]
            batch_counter = Counter(batches)
            max_same = max(batch_counter.values())
            same_batch_counts.append(max_same)
        
        same_batch_counts = np.array(same_batch_counts)
        
        # Monte Carlo comparison
        mc_counts = []
        for _ in range(3000):
            sim = np.sort(np.random.choice(range(1, max_number + 1),
                          size=n_per_draw, replace=False))
            batches = [batch_fn(n) for n in sim]
            mc_counts.append(max(Counter(batches).values()))
        
        mc_counts = np.array(mc_counts)
        
        obs_mean = np.mean(same_batch_counts)
        mc_mean = np.mean(mc_counts)
        
        # Distribution comparison
        for threshold in range(2, n_per_draw + 1):
            obs_rate = np.mean(same_batch_counts >= threshold)
            mc_rate = np.mean(mc_counts >= threshold)
            if mc_rate > 0.001:
                ratio = obs_rate / mc_rate
                flag = "⚠️" if abs(ratio - 1) > 0.15 and obs_rate > 0.01 else "  "
                print(f"    {flag} ≥{threshold} from same batch: observed={obs_rate:.1%}, "
                      f"expected={mc_rate:.1%}, ratio={ratio:.2f}")
    
    # Specific test: do balls with same LAST DIGIT appear together?
    print(f"\n  Same-last-digit co-occurrence:")
    same_unit_count = 0
    for draw in draws:
        units = [n % 10 for n in draw]
        if len(units) != len(set(units)):  # Any duplicate last digits
            same_unit_count += 1
    
    mc_same_unit = []
    for _ in range(5000):
        sim = np.random.choice(range(1, max_number + 1), size=n_per_draw, replace=False)
        units = [n % 10 for n in sim]
        mc_same_unit.append(len(units) != len(set(units)))
    
    obs_rate = same_unit_count / n_draws
    mc_rate = np.mean(mc_same_unit)
    p_val = stats.binomtest(same_unit_count, n_draws, mc_rate).pvalue
    
    print(f"    Draws with shared last digit: {same_unit_count}/{n_draws} ({obs_rate:.1%})")
    print(f"    Expected: {mc_rate:.1%}")
    print(f"    p-value: {p_val:.6f}")
    print(f"    {'⚠️  BATCH EFFECT ON UNITS DIGIT' if p_val < 0.05 else '✓ No units digit batch effect'}")
    
    return {'same_unit_p': p_val}


# =====================================================================
# P3: INK/PAINT WEIGHT EFFECT
# =====================================================================

def method_p03_ink_weight(draws, max_number=49):
    """
    WHAT IT TESTS:
      Do single-digit numbers (1-9) behave differently from double-digit (10-49)?
      Do numbers with more ink (wider digits like 8,0) differ from narrow (1,7)?
      
    WHY IT MATTERS:
      More ink/paint = slightly more weight. A ball labeled "48" has more
      paint than "1". In a sensitive machine, this could matter.
      
    TESTS:
      1. Single digit (1-9) vs double digit (10-49) frequency
      2. Total ink estimate based on digit shapes
      3. Sum of digits correlation
    """
    print("\n" + "=" * 70)
    print("METHOD P3: INK/PAINT WEIGHT PROXY ANALYSIS")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    all_numbers = draws.flatten()
    
    # Test 1: Single vs double digit
    single_digit = np.sum(all_numbers <= 9)
    total = len(all_numbers)
    expected_single_rate = 9 / max_number
    observed_single_rate = single_digit / total
    p_single = stats.binomtest(single_digit, total, expected_single_rate).pvalue
    
    print(f"\n  Test 1: Single-digit (1-9) vs double-digit (10-{max_number})")
    print(f"    Single-digit frequency: {observed_single_rate:.4f} (expected: {expected_single_rate:.4f})")
    print(f"    p-value: {p_single:.6f}")
    print(f"    {'⚠️  DIGIT COUNT BIAS' if p_single < 0.05 else '✓ No digit count bias'}")
    
    # Test 2: Ink weight proxy
    # Approximate ink coverage per digit (relative scale)
    # Wide digits (0,8,6,9) use more ink than narrow (1,7)
    ink_weight = {
        '0': 6, '1': 2, '2': 5, '3': 5, '4': 4,
        '5': 5, '6': 6, '7': 3, '8': 7, '9': 6
    }
    
    def ball_ink(n):
        return sum(ink_weight[d] for d in str(n))
    
    # Correlation between ink weight and frequency
    frequencies = np.bincount(all_numbers, minlength=max_number + 1)[1:]
    ink_weights = np.array([ball_ink(n) for n in range(1, max_number + 1)])
    
    corr, p_ink = spearmanr(ink_weights, frequencies)
    
    print(f"\n  Test 2: Ink weight proxy vs frequency")
    print(f"    Correlation: ρ={corr:+.4f}, p={p_ink:.6f}")
    print(f"    {'⚠️  INK WEIGHT CORRELATES WITH FREQUENCY' if p_ink < 0.05 else '✓ No ink weight effect'}")
    
    # Test 3: Sum of digits
    def digit_sum(n):
        return sum(int(d) for d in str(n))
    
    digit_sums = np.array([digit_sum(n) for n in range(1, max_number + 1)])
    corr_ds, p_ds = spearmanr(digit_sums, frequencies)
    
    print(f"\n  Test 3: Digit sum vs frequency")
    print(f"    Correlation: ρ={corr_ds:+.4f}, p={p_ds:.6f}")
    print(f"    {'⚠️  DIGIT SUM EFFECT' if p_ds < 0.05 else '✓ No digit sum effect'}")
    
    # Test 4: Number magnitude vs frequency (linear weight proxy)
    # Heavier numbers printed with more material?
    corr_mag, p_mag = spearmanr(range(1, max_number + 1), frequencies)
    
    print(f"\n  Test 4: Number magnitude vs frequency")
    print(f"    Correlation: ρ={corr_mag:+.4f}, p={p_mag:.6f}")
    print(f"    {'⚠️  MAGNITUDE BIAS' if p_mag < 0.05 else '✓ No magnitude bias'}")
    
    # Test 5: Group numbers by ink weight and compare
    print(f"\n  Test 5: Frequency by ink weight group:")
    light_nums = [n for n in range(1, max_number + 1) if ball_ink(n) <= 5]
    medium_nums = [n for n in range(1, max_number + 1) if 5 < ball_ink(n) <= 9]
    heavy_nums = [n for n in range(1, max_number + 1) if ball_ink(n) > 9]
    
    for name, nums in [("Light ink", light_nums), ("Medium ink", medium_nums), ("Heavy ink", heavy_nums)]:
        if nums:
            freqs = [frequencies[n-1] for n in nums]
            mean_freq = np.mean(freqs)
            expected_freq = total / max_number
            print(f"    {name:>11s} ({len(nums):2d} numbers): mean freq={mean_freq:.1f} "
                  f"(expected {expected_freq:.1f})")
    
    return {'ink_p': p_ink, 'digit_sum_p': p_ds, 'magnitude_p': p_mag}


# =====================================================================
# P4: PHYSICAL ADJACENCY / MACHINE LAYOUT
# =====================================================================

def method_p04_adjacency(draws, max_number=49):
    """
    WHAT IT TESTS:
      Do numerically adjacent balls appear together more than expected?
      Do balls that might be PHYSICALLY adjacent in the machine co-occur?
      
    WHY IT MATTERS:
      In many machines, balls are loaded in order or arranged spatially.
      If mixing is imperfect, neighbors in the machine are drawn together.
      
    LAYOUTS TESTED:
      1. Sequential adjacency (n, n+1, n+2)
      2. Circular layout (1 next to 49)
      3. Grid layout (7×7 grid arrangement)
      4. Two-row layout (odds on one side, evens on other)
    """
    print("\n" + "=" * 70)
    print("METHOD P4: PHYSICAL ADJACENCY / MACHINE LAYOUT")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    
    # Layout 1: Sequential (already tested in Method 13, but repeat for completeness)
    print(f"\n  Layout 1: Sequential adjacency")
    seq_adj_count = sum(1 for draw in draws 
                        for i in range(len(draw)-1) 
                        if abs(draw[i] - draw[i+1]) <= 1)  # Assuming sorted
    # Better: check all pairs
    total_adj_pairs = 0
    for draw in draws:
        for i in range(len(draw)):
            for j in range(i+1, len(draw)):
                if abs(draw[i] - draw[j]) == 1:
                    total_adj_pairs += 1
    
    # Monte Carlo
    mc_adj = []
    for _ in range(5000):
        sim = np.sort(np.random.choice(range(1, max_number+1), size=n_per_draw, replace=False))
        adj = sum(1 for i in range(len(sim)) for j in range(i+1, len(sim)) if abs(sim[i]-sim[j]) == 1)
        mc_adj.append(adj)
    mc_adj = np.array(mc_adj)
    
    mean_adj = total_adj_pairs / n_draws
    mc_mean = np.mean(mc_adj)
    z = (mean_adj - mc_mean) / np.std(mc_adj) * np.sqrt(n_draws) if np.std(mc_adj) > 0 else 0
    print(f"    Mean adjacent pairs per draw: {mean_adj:.3f} (expected: {mc_mean:.3f})")
    print(f"    Z-score: {z:.2f}")
    
    # Layout 2: Circular (1 is adjacent to 49)
    print(f"\n  Layout 2: Circular adjacency (1 next to {max_number})")
    circ_adj_pairs = 0
    for draw in draws:
        for i in range(len(draw)):
            for j in range(i+1, len(draw)):
                diff = abs(draw[i] - draw[j])
                if diff == 1 or diff == max_number - 1:
                    circ_adj_pairs += 1
    
    circ_mean = circ_adj_pairs / n_draws
    print(f"    Mean circular-adjacent pairs per draw: {circ_mean:.3f}")
    
    # Layout 3: Modular adjacency (same position in repeating pattern)
    print(f"\n  Layout 3: Modular adjacency (mod 7, like rows in a grid)")
    for modulus in [5, 7, 10]:
        mod_same = 0
        for draw in draws:
            mods = [n % modulus for n in draw]
            if len(mods) != len(set(mods)):
                mod_same += 1
        
        mc_mod = []
        for _ in range(3000):
            sim = np.random.choice(range(1, max_number+1), size=n_per_draw, replace=False)
            mods = [n % modulus for n in sim]
            mc_mod.append(len(mods) != len(set(mods)))
        
        obs_rate = mod_same / n_draws
        mc_rate = np.mean(mc_mod)
        if mc_rate > 0:
            ratio = obs_rate / mc_rate
            flag = "⚠️" if abs(ratio - 1) > 0.1 else "  "
            print(f"    {flag} Mod {modulus}: same-mod rate={obs_rate:.1%} "
                  f"(expected {mc_rate:.1%}, ratio={ratio:.2f})")
    
    # Layout 4: Spatial proximity score
    print(f"\n  Layout 4: Overall spatial clustering score")
    # Use mean absolute difference between drawn numbers as proxy
    mad_observed = [np.mean([abs(draw[i]-draw[j]) 
                    for i in range(len(draw)) for j in range(i+1, len(draw))]) 
                    for draw in draws]
    
    mc_mad = []
    for _ in range(5000):
        sim = np.sort(np.random.choice(range(1, max_number+1), size=n_per_draw, replace=False))
        mc_mad.append(np.mean([abs(sim[i]-sim[j]) 
                      for i in range(len(sim)) for j in range(i+1, len(sim))]))
    
    ks_stat, ks_p = ks_2samp(mad_observed, mc_mad)
    print(f"    Mean absolute difference: observed={np.mean(mad_observed):.2f}, "
          f"expected={np.mean(mc_mad):.2f}")
    print(f"    KS test: statistic={ks_stat:.4f}, p={ks_p:.6f}")
    print(f"    {'⚠️  SPATIAL CLUSTERING DETECTED' if ks_p < 0.05 else '✓ No spatial clustering'}")
    
    return {'ks_p': ks_p}


# =====================================================================
# P5: BALL WEAR / AGE EFFECTS
# =====================================================================

def method_p05_wear(draws, dates, max_number=49):
    """
    WHAT IT TESTS:
      Does a ball's frequency change over time? (wearing out / getting smoother)
      Do balls that appear more often become MORE likely (positive feedback)?
      
    WHY IT MATTERS:
      A frequently drawn ball gets handled more, potentially changing its
      surface properties. Worn balls might slide differently.
    """
    print("\n" + "=" * 70)
    print("METHOD P5: BALL WEAR / CUMULATIVE USE EFFECTS")
    print("=" * 70)
    
    n_draws = len(draws)
    
    # Track cumulative frequency and test for acceleration/deceleration
    print(f"\n  Testing if ball frequency accelerates (positive feedback) or decelerates:")
    
    accelerating = []
    decelerating = []
    
    for num in range(1, max_number + 1):
        # Split into first half and second half
        half = n_draws // 2
        first_half = sum(1 for draw in draws[:half] if num in draw)
        second_half = sum(1 for draw in draws[half:] if num in draw)
        
        rate_1 = first_half / half
        rate_2 = second_half / (n_draws - half)
        
        # Binomial test for change
        p = stats.binomtest(second_half, n_draws - half, rate_1).pvalue if rate_1 > 0 else 1.0
        
        if p < 0.05:
            if rate_2 > rate_1:
                accelerating.append((num, rate_1, rate_2, p))
            else:
                decelerating.append((num, rate_1, rate_2, p))
    
    if accelerating:
        print(f"\n  ⚠️  Numbers becoming MORE frequent (wearing in?):")
        for num, r1, r2, p in accelerating:
            print(f"    Number {num:2d}: {r1:.1%} → {r2:.1%} (p={p:.4f})")
    
    if decelerating:
        print(f"\n  ⚠️  Numbers becoming LESS frequent (wearing out?):")
        for num, r1, r2, p in decelerating:
            print(f"    Number {num:2d}: {r1:.1%} → {r2:.1%} (p={p:.4f})")
    
    if not accelerating and not decelerating:
        print(f"    ✓ No significant frequency changes over time")
    
    # Cumulative draw count correlation
    print(f"\n  Positive feedback test (does being drawn make future drawing more likely?):")
    
    feedback_scores = []
    for num in range(1, max_number + 1):
        presence = [1 if num in draw else 0 for draw in draws]
        cumulative = np.cumsum(presence)
        # Correlation between cumulative count and next appearance
        if sum(presence) > 10:
            # For each draw, does cumulative count predict next appearance?
            x = cumulative[:-1]
            y = presence[1:]
            corr, p = pearsonr(x, y)
            feedback_scores.append((num, corr, p))
    
    sig_feedback = [(n, c, p) for n, c, p in feedback_scores if p < 0.05]
    if sig_feedback:
        print(f"  ⚠️  {len(sig_feedback)} numbers show feedback effects:")
        for num, corr, p in sorted(sig_feedback, key=lambda x: x[2])[:10]:
            direction = "positive (more begets more)" if corr > 0 else "negative (more begets less)"
            print(f"    Number {num:2d}: r={corr:+.4f}, p={p:.4f} — {direction}")
    else:
        print(f"    ✓ No feedback effects detected")
    
    return {'n_accelerating': len(accelerating), 'n_decelerating': len(decelerating),
            'n_feedback': len(sig_feedback)}


# =====================================================================
# P6: MIXING QUALITY ANALYSIS
# =====================================================================

def method_p06_mixing(draws, max_number=49):
    """
    WHAT IT TESTS:
      How well does the machine mix the balls between draws?
      
    WHY IT MATTERS:
      If mixing is incomplete, balls from the previous draw may
      still be clustered together, creating carryover effects.
      
    TESTS:
      1. Overlap between consecutive draws
      2. Do numbers from draw N appear in draw N+1 more than expected?
      3. Anti-persistence (does the machine deliberately avoid repeats?)
    """
    print("\n" + "=" * 70)
    print("METHOD P6: MIXING QUALITY ANALYSIS")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    
    # Test 1: Overlap between consecutive draws
    overlaps = []
    for i in range(n_draws - 1):
        common = len(set(draws[i]) & set(draws[i + 1]))
        overlaps.append(common)
    
    overlaps = np.array(overlaps)
    
    # Monte Carlo expected overlap
    mc_overlaps = []
    for _ in range(10000):
        d1 = set(np.random.choice(range(1, max_number + 1), size=n_per_draw, replace=False))
        d2 = set(np.random.choice(range(1, max_number + 1), size=n_per_draw, replace=False))
        mc_overlaps.append(len(d1 & d2))
    mc_overlaps = np.array(mc_overlaps)
    
    print(f"\n  Test 1: Consecutive draw overlap")
    print(f"    Mean overlap: {np.mean(overlaps):.3f} numbers")
    print(f"    Expected overlap: {np.mean(mc_overlaps):.3f} numbers")
    
    ks_stat, ks_p = ks_2samp(overlaps, mc_overlaps)
    print(f"    KS test: statistic={ks_stat:.4f}, p={ks_p:.6f}")
    
    # Distribution of overlaps
    print(f"\n    Overlap distribution:")
    for k in range(n_per_draw + 1):
        obs = np.mean(overlaps == k)
        exp = np.mean(mc_overlaps == k)
        flag = "⚠️" if abs(obs - exp) > 0.03 else "  "
        print(f"      {flag} {k} shared: observed={obs:.1%}, expected={exp:.1%}")
    
    # Test 2: Gap-1 repeat rate per number
    print(f"\n  Test 2: Per-number repeat rate (appears in draw N AND N+1)")
    high_repeat_nums = []
    low_repeat_nums = []
    expected_rate = (n_per_draw / max_number) ** 2  # Independence assumption
    
    for num in range(1, max_number + 1):
        presence = [num in draw for draw in draws]
        repeats = sum(1 for i in range(len(presence)-1) if presence[i] and presence[i+1])
        expected_repeats = sum(1 for i in range(len(presence)-1) if presence[i]) * (n_per_draw / max_number)
        
        if expected_repeats > 3:
            ratio = repeats / expected_repeats if expected_repeats > 0 else 0
            p = stats.binomtest(repeats, 
                               sum(1 for i in range(len(presence)-1) if presence[i]),
                               n_per_draw / max_number).pvalue
            if p < 0.05 and ratio > 1.3:
                high_repeat_nums.append((num, repeats, expected_repeats, ratio, p))
            elif p < 0.05 and ratio < 0.7:
                low_repeat_nums.append((num, repeats, expected_repeats, ratio, p))
    
    if high_repeat_nums:
        print(f"\n    ⚠️  Numbers that REPEAT more than expected (poor mixing):")
        for num, obs, exp, ratio, p in high_repeat_nums:
            print(f"      Number {num:2d}: {obs} repeats (expected {exp:.1f}), ratio={ratio:.2f}, p={p:.4f}")
    
    if low_repeat_nums:
        print(f"\n    ⚠️  Numbers that AVOID repeating (over-mixing or anti-persistence):")
        for num, obs, exp, ratio, p in low_repeat_nums:
            print(f"      Number {num:2d}: {obs} repeats (expected {exp:.1f}), ratio={ratio:.2f}, p={p:.4f}")
    
    if not high_repeat_nums and not low_repeat_nums:
        print(f"    ✓ All numbers have normal repeat rates")
    
    # Test 3: Does overlap decrease/increase over time?
    print(f"\n  Test 3: Overlap trend over time")
    idx = np.arange(len(overlaps))
    slope, _, _, p_trend, _ = stats.linregress(idx, overlaps)
    print(f"    Slope: {slope:.6f} per draw")
    print(f"    p-value: {p_trend:.6f}")
    print(f"    {'⚠️  MIXING QUALITY CHANGING OVER TIME' if p_trend < 0.05 else '✓ Mixing quality stable'}")
    
    return {'overlap_ks_p': ks_p, 'n_high_repeat': len(high_repeat_nums),
            'n_low_repeat': len(low_repeat_nums)}


# =====================================================================
# P7: BALL INTERACTION NETWORK
# =====================================================================

def method_p07_interaction_network(draws, max_number=49):
    """
    WHAT IT TESTS:
      Do certain balls "attract" or "repel" each other?
      
    WHY IT MATTERS:
      Physical interactions between balls (weight, size, surface)
      could cause certain combinations to appear together more.
      Unlike pair frequency (Method 2), this looks at the NETWORK
      structure of interactions.
      
    APPROACH:
      Build a co-occurrence matrix, then analyze its structure:
      - Are there clusters of mutually-attracting balls?
      - Are there isolated balls that avoid others?
      - Is the network structure random or structured?
    """
    print("\n" + "=" * 70)
    print("METHOD P7: BALL INTERACTION NETWORK")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    
    # Build co-occurrence matrix
    cooccurrence = np.zeros((max_number, max_number))
    for draw in draws:
        for i in range(len(draw)):
            for j in range(i + 1, len(draw)):
                cooccurrence[draw[i]-1][draw[j]-1] += 1
                cooccurrence[draw[j]-1][draw[i]-1] += 1
    
    # Expected co-occurrence
    expected = n_draws * (n_per_draw * (n_per_draw - 1)) / (max_number * (max_number - 1))
    
    # Deviation matrix
    deviation = (cooccurrence - expected) / expected * 100
    np.fill_diagonal(deviation, 0)
    
    # Find strongest attractions and repulsions
    attractions = []
    repulsions = []
    for i in range(max_number):
        for j in range(i + 1, max_number):
            dev = deviation[i][j]
            if abs(dev) > 50:  # More than 50% deviation
                if dev > 0:
                    attractions.append((i+1, j+1, cooccurrence[i][j], expected, dev))
                else:
                    repulsions.append((i+1, j+1, cooccurrence[i][j], expected, dev))
    
    print(f"\n  Expected co-occurrence per pair: {expected:.2f}")
    
    if attractions:
        print(f"\n  ⚠️  Strongest ATTRACTIONS (co-occur more than expected):")
        for a, b, obs, exp, dev in sorted(attractions, key=lambda x: x[4], reverse=True)[:10]:
            print(f"    Balls {a:2d}-{b:2d}: {obs:.0f} times (expected {exp:.1f}, +{dev:.0f}%)")
    
    if repulsions:
        print(f"\n  ⚠️  Strongest REPULSIONS (avoid each other):")
        for a, b, obs, exp, dev in sorted(repulsions, key=lambda x: x[4])[:10]:
            print(f"    Balls {a:2d}-{b:2d}: {obs:.0f} times (expected {exp:.1f}, {dev:.0f}%)")
    
    # Network clustering coefficient
    # For each ball, what fraction of its "neighbors" also co-occur together?
    print(f"\n  Network clustering analysis:")
    threshold = expected * 1.3  # 30% above expected = "connected"
    adjacency = (cooccurrence > threshold).astype(int)
    np.fill_diagonal(adjacency, 0)
    
    degrees = adjacency.sum(axis=1)
    print(f"    Mean connections per ball (30% above expected): {np.mean(degrees):.1f}")
    print(f"    Max connections: ball {np.argmax(degrees)+1} ({int(np.max(degrees))} connections)")
    print(f"    Min connections: ball {np.argmin(degrees)+1} ({int(np.min(degrees))} connections)")
    
    # Is degree distribution random?
    expected_degree = np.mean(degrees)
    chi2, p = chisquare(np.bincount(degrees.astype(int), minlength=int(np.max(degrees))+1)[1:])
    print(f"    Degree distribution χ²={chi2:.2f}, p={p:.4f}")
    print(f"    {'⚠️  NON-RANDOM NETWORK STRUCTURE' if p < 0.05 else '✓ Random network structure'}")
    
    return {'n_attractions': len(attractions), 'n_repulsions': len(repulsions)}


# =====================================================================
# P8: DRAW-WITHIN-DRAW SEQUENTIAL ANALYSIS
# =====================================================================

def method_p08_within_draw(draws, max_number=49):
    """
    WHAT IT TESTS:
      If the data preserves draw ORDER (not sorted), does the
      position of each ball in the sequence reveal machine bias?
      
      Even if sorted, tests if the SPACING pattern within a draw
      (gaps between consecutive numbers) reveals machine geometry.
    """
    print("\n" + "=" * 70)
    print("METHOD P8: WITHIN-DRAW SEQUENTIAL PATTERNS")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    
    # Gap patterns within each draw
    all_gap_sequences = []
    for draw in draws:
        sorted_draw = np.sort(draw)
        gaps = tuple(np.diff(sorted_draw))
        all_gap_sequences.append(gaps)
    
    # Are certain gap patterns repeated?
    gap_counter = Counter(all_gap_sequences)
    repeated_patterns = {k: v for k, v in gap_counter.items() if v > 1}
    
    print(f"\n  Unique gap patterns: {len(gap_counter)}")
    print(f"  Repeated gap patterns: {len(repeated_patterns)}")
    
    if repeated_patterns:
        print(f"\n  Most common gap patterns (top 10):")
        for pattern, count in gap_counter.most_common(10):
            print(f"    Gaps {pattern}: {count} times")
    
    # Monte Carlo comparison
    mc_unique = []
    for _ in range(1000):
        sim_gaps = []
        for _ in range(n_draws):
            sim = np.sort(np.random.choice(range(1, max_number+1), 
                          size=n_per_draw, replace=False))
            sim_gaps.append(tuple(np.diff(sim)))
        mc_unique.append(len(set(sim_gaps)))
    
    mc_mean = np.mean(mc_unique)
    z = (len(gap_counter) - mc_mean) / np.std(mc_unique) if np.std(mc_unique) > 0 else 0
    print(f"\n  Unique patterns: observed={len(gap_counter)}, expected={mc_mean:.1f}")
    print(f"  Z-score: {z:.2f}")
    print(f"  {'⚠️  ABNORMAL GAP PATTERN DIVERSITY' if abs(z) > 2 else '✓ Normal gap pattern diversity'}")
    
    # First gap vs last gap correlation
    first_gaps = [g[0] for g in all_gap_sequences]
    last_gaps = [g[-1] for g in all_gap_sequences]
    corr, p = spearmanr(first_gaps, last_gaps)
    print(f"\n  First gap vs last gap correlation: ρ={corr:+.4f}, p={p:.6f}")
    print(f"  {'⚠️  MACHINE GEOMETRY EFFECT' if p < 0.05 else '✓ No first/last gap correlation'}")
    
    # Average gap profile
    print(f"\n  Average gap profile (machine geometry fingerprint):")
    avg_gaps = np.mean([list(g) for g in all_gap_sequences], axis=0)
    mc_avg_gaps = []
    for _ in range(5000):
        sim = np.sort(np.random.choice(range(1, max_number+1), 
                      size=n_per_draw, replace=False))
        mc_avg_gaps.append(np.diff(sim))
    mc_avg = np.mean(mc_avg_gaps, axis=0)
    
    for i in range(len(avg_gaps)):
        dev = (avg_gaps[i] - mc_avg[i]) / mc_avg[i] * 100
        flag = "⚠️" if abs(dev) > 10 else "  "
        print(f"    {flag} Gap {i+1}: observed={avg_gaps[i]:.2f}, expected={mc_avg[i]:.2f} ({dev:+.1f}%)")
    
    return {'z_score': z}


# =====================================================================
# P9: MACHINE WARM-UP / COOL-DOWN EFFECT
# =====================================================================

def method_p09_warmup(draws, dates, max_number=49):
    """
    WHAT IT TESTS:
      Are early draws in a session different from later draws?
      Does the machine behave differently when "cold" vs "warmed up"?
      
    WHY IT MATTERS:
      Physical machines have temperature-dependent behavior.
      Lubricants, ball materials, and air pressure change with use.
    """
    print("\n" + "=" * 70)
    print("METHOD P9: MACHINE WARM-UP / SESSION EFFECTS")
    print("=" * 70)
    
    if dates is None:
        print("  ⚠️  No dates available, skipping session analysis")
        return {}
    
    # Identify sessions (draws on the same day or within 1 day = same session)
    n_draws = len(draws)
    draw_sums = draws.sum(axis=1)
    
    # Since we likely have 1 draw per session, look at day-of-week patterns
    # First draw of the week vs last
    day_of_week = dates.dt.dayofweek
    
    # Alternative: look at sequential position within the dataset
    # First N draws vs last N draws
    segment_size = n_draws // 5
    
    print(f"\n  Comparing first {segment_size} draws vs last {segment_size} draws:")
    first_sums = draw_sums[:segment_size]
    last_sums = draw_sums[-segment_size:]
    
    ks, p = ks_2samp(first_sums, last_sums)
    print(f"    First segment: mean={np.mean(first_sums):.1f}, std={np.std(first_sums):.1f}")
    print(f"    Last segment:  mean={np.mean(last_sums):.1f}, std={np.std(last_sums):.1f}")
    print(f"    KS test: p={p:.6f}")
    print(f"    {'⚠️  MACHINE BEHAVIOR CHANGED' if p < 0.05 else '✓ Consistent behavior'}")
    
    # Compare number frequencies: first vs last segment
    first_flat = draws[:segment_size].flatten()
    last_flat = draws[-segment_size:].flatten()
    first_freq = np.bincount(first_flat, minlength=max_number+1)[1:].astype(float)
    last_freq = np.bincount(last_flat, minlength=max_number+1)[1:].astype(float)
    first_freq /= first_freq.sum()
    last_freq /= last_freq.sum()
    
    # Numbers with biggest shift
    shifts = last_freq - first_freq
    biggest_increase = np.argsort(shifts)[-3:] + 1
    biggest_decrease = np.argsort(shifts)[:3] + 1
    
    print(f"\n  Numbers gaining frequency over time: {list(biggest_increase)}")
    print(f"  Numbers losing frequency over time:  {list(biggest_decrease)}")
    
    # 5-segment trend analysis
    print(f"\n  5-segment trend (is the machine drifting?):")
    for seg in range(5):
        start = seg * segment_size
        end = start + segment_size
        s = draw_sums[start:end]
        print(f"    Segment {seg+1} (draws {start}-{end}): "
              f"mean={np.mean(s):.1f}, std={np.std(s):.1f}")
    
    return {'ks_p': p}


# =====================================================================
# P10: NUMBER MODULAR ARITHMETIC PATTERNS
# =====================================================================

def method_p10_modular(draws, max_number=49):
    """
    WHAT IT TESTS:
      Do drawn numbers have patterns in modular arithmetic?
      (e.g., all numbers ≡ 3 mod 7 more often than expected)
      
    WHY IT MATTERS:
      If the machine has N pockets/slots arranged in a circle,
      numbers assigned to certain positions would show modular patterns.
      This can reveal the machine's physical geometry.
    """
    print("\n" + "=" * 70)
    print("METHOD P10: MODULAR ARITHMETIC PATTERNS (Machine Geometry)")
    print("=" * 70)
    
    n_draws = len(draws)
    n_per_draw = draws.shape[1]
    all_numbers = draws.flatten()
    
    # Test various moduli (corresponding to possible machine geometries)
    for mod in [2, 3, 5, 7, 8, 10, 12]:
        residues = all_numbers % mod
        obs = np.bincount(residues, minlength=mod)
        
        # Expected (not perfectly uniform because max_number may not be divisible by mod)
        expected = np.zeros(mod)
        for n in range(1, max_number + 1):
            expected[n % mod] += 1
        expected = expected / expected.sum() * len(all_numbers)
        
        chi2, p = chisquare(obs, expected)
        flag = "⚠️" if p < 0.05 else "  "
        print(f"  {flag} Mod {mod:2d}: χ²={chi2:8.2f}, p={p:.6f}")
        
        if p < 0.05:
            print(f"         Residue distribution: {dict(enumerate(obs))}")
            print(f"         Expected:             {dict(enumerate(expected.astype(int)))}")
    
    # Within-draw modular analysis
    print(f"\n  Within-draw modular concentration:")
    for mod in [2, 3, 5, 7]:
        concentrations = []
        for draw in draws:
            residues = [n % mod for n in draw]
            max_same = max(Counter(residues).values())
            concentrations.append(max_same)
        
        mc_conc = []
        for _ in range(3000):
            sim = np.random.choice(range(1, max_number+1), size=n_per_draw, replace=False)
            mc_conc.append(max(Counter([n % mod for n in sim]).values()))
        
        obs_mean = np.mean(concentrations)
        mc_mean = np.mean(mc_conc)
        z = (obs_mean - mc_mean) / np.std(mc_conc) if np.std(mc_conc) > 0 else 0
        flag = "⚠️" if abs(z) > 2 else "  "
        print(f"    {flag} Mod {mod}: mean max-same-residue={obs_mean:.2f} (expected {mc_mean:.2f}), Z={z:.2f}")
    
    return {}


# =====================================================================
# SUMMARY
# =====================================================================

def print_physical_summary(results):
    print("\n\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "  PHYSICAL MACHINE BIAS SUMMARY  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    print(f"""
  Physical machine inconsistencies tested:
  
  P1.  Positional bias          - Does position in draw matter?
  P2.  Batch effects            - Do similar numbers cluster?
  P3.  Ink/paint weight         - Does physical weight of markings matter?
  P4.  Physical adjacency       - Are neighbors in machine drawn together?
  P5.  Ball wear                - Do frequently used balls change behavior?
  P6.  Mixing quality           - Does the machine mix well between draws?
  P7.  Interaction network      - Do certain balls attract/repel each other?
  P8.  Within-draw patterns     - Does gap structure reveal geometry?
  P9.  Warm-up effects          - Does machine behavior change during use?
  P10. Modular arithmetic       - Does machine geometry create mod patterns?
    """)


# =====================================================================
# MAIN
# =====================================================================

def run_physical_analysis(filepath='Data.csv', sep=';',
                           date_col='date', number_cols=None,
                           max_number=49, date_format='%d/%m/%Y'):
    
    print("╔" + "═" * 68 + "╗")
    print("║" + "  PHYSICAL MACHINE BIAS DETECTION  ".center(68) + "║")
    print("║" + "  10 Methods for Mechanical Inconsistencies  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    print("\n📊 LOADING DATA")
    draws, dates, df = load_data(filepath, sep, date_col, number_cols, date_format)
    
    if number_cols is None:
        number_cols = [c for c in df.columns if c != date_col]
    
    results = {}
    
    results['p01'] = method_p01_positional(draws, max_number)
    results['p02'] = method_p02_batch(draws, max_number)
    results['p03'] = method_p03_ink_weight(draws, max_number)
    results['p04'] = method_p04_adjacency(draws, max_number)
    results['p05'] = method_p05_wear(draws, dates, max_number)
    results['p06'] = method_p06_mixing(draws, max_number)
    results['p07'] = method_p07_interaction_network(draws, max_number)
    results['p08'] = method_p08_within_draw(draws, max_number)
    results['p09'] = method_p09_warmup(draws, dates, max_number)
    results['p10'] = method_p10_modular(draws, max_number)
    
    print_physical_summary(results)
    
    return results


if __name__ == "__main__":
    results = run_physical_analysis(
        filepath='Data.csv',
        sep=';',
        date_col='date',
        number_cols=['n1', 'n2', 'n3', 'n4', 'n5'],
        max_number=49,
        date_format='%d/%m/%Y'
    )
