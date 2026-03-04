"""
============================================================================
HMM PIPELINE FOR DETECTING PATTERNS IN A RIGGED LOTTERY
============================================================================

This pipeline uses Hidden Markov Models to detect hidden regimes
(e.g., "fair" vs "rigged" states) in lottery draw data.

PIPELINE OVERVIEW:
  Step 1: Load & preprocess the data
  Step 2: Engineer features from raw draw numbers
  Step 3: Classical randomness tests (baseline sanity checks)
  Step 4: Fit HMMs with different numbers of hidden states
  Step 5: Model selection (BIC/AIC to pick best number of states)
  Step 6: Decode the hidden states (Viterbi algorithm)
  Step 7: Analyze what each regime looks like
  Step 8: Visualize everything

============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from hmmlearn import hmm
from scipy import stats
from scipy.stats import chi2_contingency, entropy
from itertools import product as itertools_product
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# =====================================================================
# STEP 0: GENERATE SYNTHETIC "RIGGED" LOTTERY DATA FOR DEMONSTRATION
# =====================================================================
# In practice, you'd load your real dataset here.
# This simulation creates a lottery that switches between fair and rigged.

def generate_rigged_lottery(n_draws=1000, n_numbers=6, max_number=49):
    """
    Simulates a rigged lottery with two hidden regimes:
      - Fair:   numbers drawn uniformly at random
      - Rigged: certain "favored" numbers have higher probability
    
    The regime switches according to a Markov process.
    """
    # Markov transition matrix for regimes
    # Fair->Fair=0.95, Fair->Rigged=0.05, Rigged->Rigged=0.90, Rigged->Fair=0.10
    transition_matrix = np.array([
        [0.95, 0.05],  # From Fair
        [0.10, 0.90],  # From Rigged
    ])
    
    # Favored numbers in rigged state (higher probability for these)
    favored_numbers = [7, 13, 21, 33, 42]
    favor_boost = 3.0  # These numbers are 3x more likely when rigged
    
    # Build rigged probability distribution
    rigged_probs = np.ones(max_number) / max_number
    for num in favored_numbers:
        rigged_probs[num - 1] *= favor_boost
    rigged_probs /= rigged_probs.sum()
    
    fair_probs = np.ones(max_number) / max_number
    
    # Generate draws
    draws = []
    true_states = []
    state = 0  # Start in Fair state
    
    for i in range(n_draws):
        true_states.append(state)
        
        if state == 0:  # Fair
            draw = np.random.choice(range(1, max_number + 1), size=n_numbers,
                                     replace=False, p=fair_probs)
        else:  # Rigged
            # Draw with replacement-aware approach for rigged probabilities
            draw = []
            probs = rigged_probs.copy()
            for _ in range(n_numbers):
                chosen = np.random.choice(range(1, max_number + 1), p=probs)
                draw.append(chosen)
                # Remove chosen number and renormalize
                probs[chosen - 1] = 0
                if probs.sum() > 0:
                    probs /= probs.sum()
            draw = np.array(draw)
        
        draws.append(np.sort(draw))
        
        # Transition to next state
        state = np.random.choice([0, 1], p=transition_matrix[state])
    
    return np.array(draws), np.array(true_states), favored_numbers


# =====================================================================
# STEP 1: FEATURE ENGINEERING
# =====================================================================
# Raw lottery numbers aren't great direct inputs for an HMM.
# We extract STATISTICAL FEATURES from each draw.

def engineer_features(draws, max_number=49):
    """
    Extract meaningful features from each lottery draw.
    These features capture properties that differ between fair and rigged draws.
    
    Features:
      1. mean        - Average of drawn numbers
      2. std         - Spread of drawn numbers  
      3. range_val   - Max minus min
      4. sum_val     - Sum of all numbers
      5. median      - Median of drawn numbers
      6. skewness    - Asymmetry of the draw
      7. min_gap     - Smallest gap between consecutive sorted numbers
      8. max_gap     - Largest gap between consecutive sorted numbers
      9. mean_gap    - Average gap between consecutive sorted numbers
      10. low_count  - How many numbers fall in lower third (1-16)
      11. mid_count  - How many numbers fall in middle third (17-33)
      12. high_count - How many numbers fall in upper third (34-49)
      13. even_count - How many even numbers
      14. hot_count  - Count of commonly "hot" numbers in the draw
    """
    features = []
    
    for draw in draws:
        sorted_draw = np.sort(draw)
        gaps = np.diff(sorted_draw)
        
        feat = {
            'mean': np.mean(draw),
            'std': np.std(draw),
            'range_val': np.max(draw) - np.min(draw),
            'sum_val': np.sum(draw),
            'median': np.median(draw),
            'skewness': stats.skew(draw),
            'min_gap': np.min(gaps) if len(gaps) > 0 else 0,
            'max_gap': np.max(gaps) if len(gaps) > 0 else 0,
            'mean_gap': np.mean(gaps) if len(gaps) > 0 else 0,
            'low_count': np.sum(draw <= max_number // 3),
            'mid_count': np.sum((draw > max_number // 3) & (draw <= 2 * max_number // 3)),
            'high_count': np.sum(draw > 2 * max_number // 3),
            'even_count': np.sum(draw % 2 == 0),
        }
        features.append(feat)
    
    return pd.DataFrame(features)


# =====================================================================
# STEP 2: CLASSICAL RANDOMNESS TESTS (BASELINE)
# =====================================================================

def run_randomness_tests(draws, max_number=49):
    """
    Run standard statistical tests to check if draws are truly random.
    These are your first line of defense before the HMM.
    """
    print("=" * 70)
    print("STEP 2: CLASSICAL RANDOMNESS TESTS")
    print("=" * 70)
    
    all_numbers = draws.flatten()
    
    # --- Test 1: Chi-squared frequency test ---
    observed_freq = np.bincount(all_numbers, minlength=max_number + 1)[1:]
    expected_freq = np.full(max_number, len(all_numbers) / max_number)
    chi2, p_value = stats.chisquare(observed_freq, expected_freq)
    
    print(f"\n1. CHI-SQUARED FREQUENCY TEST")
    print(f"   Tests if all numbers appear equally often")
    print(f"   Chi² = {chi2:.2f}, p-value = {p_value:.6f}")
    print(f"   → {'⚠️  SUSPICIOUS (p < 0.05)' if p_value < 0.05 else '✓ Looks fair'}")
    
    # Show the most over/under-represented numbers
    deviation = (observed_freq - expected_freq) / expected_freq * 100
    top_over = np.argsort(deviation)[-5:][::-1] + 1
    top_under = np.argsort(deviation)[:5] + 1
    print(f"   Most overrepresented:  {list(top_over)} ({[f'+{deviation[n-1]:.1f}%' for n in top_over]})")
    print(f"   Most underrepresented: {list(top_under)} ({[f'{deviation[n-1]:.1f}%' for n in top_under]})")
    
    # --- Test 2: Serial correlation (autocorrelation) ---
    print(f"\n2. SERIAL AUTOCORRELATION TEST")
    print(f"   Tests if consecutive draws are correlated")
    
    draw_sums = draws.sum(axis=1)
    max_lag = 10
    autocorrs = []
    for lag in range(1, max_lag + 1):
        corr = np.corrcoef(draw_sums[:-lag], draw_sums[lag:])[0, 1]
        autocorrs.append(corr)
        sig = "⚠️" if abs(corr) > 2 / np.sqrt(len(draw_sums)) else "  "
        print(f"   {sig} Lag {lag:2d}: r = {corr:+.4f}")
    
    # --- Test 3: Runs test ---
    print(f"\n3. RUNS TEST")
    print(f"   Tests if the sequence of above/below median is random")
    
    median_sum = np.median(draw_sums)
    binary = (draw_sums > median_sum).astype(int)
    runs = 1 + np.sum(np.diff(binary) != 0)
    n1 = np.sum(binary == 1)
    n0 = np.sum(binary == 0)
    expected_runs = 1 + 2 * n0 * n1 / (n0 + n1)
    std_runs = np.sqrt(2 * n0 * n1 * (2 * n0 * n1 - n0 - n1) / 
                       ((n0 + n1)**2 * (n0 + n1 - 1)))
    z_runs = (runs - expected_runs) / std_runs
    p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
    
    print(f"   Observed runs: {runs}, Expected: {expected_runs:.1f}")
    print(f"   Z = {z_runs:.3f}, p-value = {p_runs:.6f}")
    print(f"   → {'⚠️  SUSPICIOUS' if p_runs < 0.05 else '✓ Looks fair'}")
    
    # --- Test 4: Entropy ---
    print(f"\n4. ENTROPY ANALYSIS")
    print(f"   Measures randomness/unpredictability of number distribution")
    
    prob_dist = observed_freq / observed_freq.sum()
    observed_entropy = entropy(prob_dist)
    max_entropy = np.log(max_number)  # Uniform distribution entropy
    entropy_ratio = observed_entropy / max_entropy
    
    print(f"   Observed entropy:  {observed_entropy:.4f}")
    print(f"   Maximum entropy:   {max_entropy:.4f}")
    print(f"   Ratio:             {entropy_ratio:.4f} (1.0 = perfectly uniform)")
    print(f"   → {'⚠️  SUSPICIOUS (< 0.99)' if entropy_ratio < 0.99 else '✓ Looks fair'}")
    
    return {
        'chi2_p': p_value,
        'autocorrs': autocorrs,
        'runs_p': p_runs,
        'entropy_ratio': entropy_ratio
    }


# =====================================================================
# STEP 3: FIT HMMs WITH DIFFERENT STATE COUNTS
# =====================================================================

def fit_hmm_models(features_df, max_states=6, n_fits=10):
    """
    Fit Gaussian HMMs with 1 to max_states hidden states.
    For each state count, we run multiple fits and keep the best (highest log-likelihood)
    because HMM fitting is sensitive to initialization.
    
    Args:
        features_df: DataFrame of engineered features
        max_states: Maximum number of hidden states to try
        n_fits: Number of random restarts per state count
    
    Returns:
        Dictionary of results for each state count
    """
    print("\n" + "=" * 70)
    print("STEP 3: FITTING HMMs WITH 1-{} HIDDEN STATES".format(max_states))
    print("=" * 70)
    
    # Standardize features (important for HMM convergence)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df.values)
    
    results = {}
    
    for n_states in range(1, max_states + 1):
        best_score = -np.inf
        best_model = None
        
        for attempt in range(n_fits):
            try:
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type="full",   # Full covariance captures feature correlations
                    n_iter=200,               # Max EM iterations
                    tol=1e-4,                 # Convergence tolerance
                    random_state=attempt * 42,
                    verbose=False
                )
                model.fit(X)
                score = model.score(X)  # Log-likelihood
                
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue
        
        if best_model is not None:
            n_params = (n_states * n_states - n_states +          # Transition matrix
                       n_states - 1 +                              # Initial state probs
                       n_states * X.shape[1] +                     # Means
                       n_states * X.shape[1] * (X.shape[1] + 1) // 2)  # Covariances
            
            n_samples = len(X)
            bic = -2 * best_score + n_params * np.log(n_samples)
            aic = -2 * best_score + 2 * n_params
            
            results[n_states] = {
                'model': best_model,
                'log_likelihood': best_score,
                'bic': bic,
                'aic': aic,
                'n_params': n_params
            }
            
            print(f"\n  States={n_states}: Log-L={best_score:>10.1f}  "
                  f"BIC={bic:>10.1f}  AIC={aic:>10.1f}  "
                  f"Params={n_params}")
    
    return results, X, scaler


# =====================================================================
# STEP 4: MODEL SELECTION
# =====================================================================

def select_best_model(results):
    """
    Select the optimal number of states using BIC (Bayesian Information Criterion).
    Lower BIC = better trade-off between fit and complexity.
    
    We also check AIC for comparison.
    """
    print("\n" + "=" * 70)
    print("STEP 4: MODEL SELECTION")
    print("=" * 70)
    
    bic_values = {k: v['bic'] for k, v in results.items()}
    aic_values = {k: v['aic'] for k, v in results.items()}
    
    best_bic_states = min(bic_values, key=bic_values.get)
    best_aic_states = min(aic_values, key=aic_values.get)
    
    print(f"\n  Best by BIC: {best_bic_states} states (BIC = {bic_values[best_bic_states]:.1f})")
    print(f"  Best by AIC: {best_aic_states} states (AIC = {aic_values[best_aic_states]:.1f})")
    
    # Compare BIC of best multi-state vs single-state
    if best_bic_states > 1 and 1 in results:
        delta_bic = results[1]['bic'] - results[best_bic_states]['bic']
        print(f"\n  ΔBIC (1-state vs {best_bic_states}-state) = {delta_bic:.1f}")
        if delta_bic > 10:
            print(f"  → STRONG evidence for {best_bic_states} hidden regimes (ΔBIC > 10)")
        elif delta_bic > 6:
            print(f"  → MODERATE evidence for {best_bic_states} hidden regimes (ΔBIC > 6)")
        elif delta_bic > 2:
            print(f"  → WEAK evidence for {best_bic_states} hidden regimes (ΔBIC > 2)")
        else:
            print(f"  → NO meaningful evidence for multiple regimes")
    
    # Use BIC-selected model
    best_n = best_bic_states
    print(f"\n  ✓ Selected model: {best_n} hidden states")
    
    return best_n, results[best_n]


# =====================================================================
# STEP 5: DECODE HIDDEN STATES & ANALYZE REGIMES
# =====================================================================

def analyze_regimes(best_model_info, X, features_df, draws, scaler):
    """
    Use the Viterbi algorithm to decode the most likely sequence of hidden states,
    then analyze what each regime looks like.
    """
    print("\n" + "=" * 70)
    print("STEP 5: REGIME ANALYSIS")
    print("=" * 70)
    
    model = best_model_info['model']
    n_states = model.n_components
    
    # --- Decode hidden states using Viterbi ---
    hidden_states = model.predict(X)
    state_probs = model.predict_proba(X)
    
    # --- Transition matrix ---
    print(f"\n  TRANSITION MATRIX:")
    print(f"  (Probability of moving from state i to state j)")
    trans_mat = model.transmat_
    header = "        " + "".join([f"  To S{j}  " for j in range(n_states)])
    print(header)
    for i in range(n_states):
        row = f"  From S{i}: " + "  ".join([f"{trans_mat[i,j]:.4f}" for j in range(n_states)])
        print(row)
    
    # --- Analyze each regime ---
    print(f"\n  REGIME CHARACTERISTICS:")
    regime_stats = {}
    
    for state in range(n_states):
        mask = hidden_states == state
        count = mask.sum()
        pct = count / len(hidden_states) * 100
        
        state_features = features_df[mask]
        state_draws = draws[mask]
        
        print(f"\n  --- STATE {state} ({count} draws, {pct:.1f}% of total) ---")
        
        # Feature summaries
        for col in features_df.columns:
            print(f"    {col:>12s}: mean={state_features[col].mean():>7.2f}  "
                  f"std={state_features[col].std():>6.2f}")
        
        # Number frequency within this regime
        all_nums = state_draws.flatten()
        freq = np.bincount(all_nums, minlength=50)[1:]
        expected = len(all_nums) / 49
        most_common = np.argsort(freq)[-5:][::-1] + 1
        least_common = np.argsort(freq)[:5] + 1
        
        print(f"    Most frequent numbers:  {list(most_common)}")
        print(f"    Least frequent numbers: {list(least_common)}")
        
        # Chi-squared within regime
        chi2, p = stats.chisquare(freq, np.full(49, expected))
        print(f"    Within-regime uniformity: χ²={chi2:.1f}, p={p:.6f}")
        
        # Average duration of stays in this state
        state_changes = np.diff(hidden_states)
        in_state = (hidden_states == state)
        runs = []
        current_run = 0
        for val in in_state:
            if val:
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)
        
        if runs:
            print(f"    Avg regime duration: {np.mean(runs):.1f} consecutive draws")
            print(f"    Max regime duration: {np.max(runs)} draws")
        
        regime_stats[state] = {
            'count': count,
            'pct': pct,
            'features': state_features.describe(),
            'most_common': most_common,
            'chi2_p': p,
            'avg_duration': np.mean(runs) if runs else 0
        }
    
    return hidden_states, state_probs, regime_stats


# =====================================================================
# STEP 6: VISUALIZATION
# =====================================================================

def create_visualizations(draws, features_df, hidden_states, state_probs,
                          results, true_states=None, max_number=49):
    """Create comprehensive visualizations of the HMM analysis."""
    
    n_states = len(np.unique(hidden_states))
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_states, 3)))
    
    fig = plt.figure(figsize=(20, 24))
    
    # --- Plot 1: Hidden states over time ---
    ax1 = fig.add_subplot(5, 2, 1)
    for state in range(n_states):
        mask = hidden_states == state
        ax1.scatter(np.where(mask)[0], [state] * mask.sum(), 
                   c=[colors[state]], alpha=0.3, s=2)
    ax1.set_xlabel('Draw Number')
    ax1.set_ylabel('Detected State')
    ax1.set_title('Detected Hidden States Over Time')
    ax1.set_yticks(range(n_states))
    ax1.set_yticklabels([f'State {i}' for i in range(n_states)])
    
    # --- Plot 2: True vs detected (if available) ---
    if true_states is not None:
        ax2 = fig.add_subplot(5, 2, 2)
        ax2.plot(true_states, alpha=0.7, label='True State', color='blue', linewidth=0.5)
        ax2.plot(hidden_states, alpha=0.7, label='Detected State', color='red', linewidth=0.5)
        ax2.set_xlabel('Draw Number')
        ax2.set_ylabel('State')
        ax2.set_title('True vs Detected States')
        ax2.legend()
    
    # --- Plot 3: State probabilities over time ---
    ax3 = fig.add_subplot(5, 2, 3)
    for state in range(n_states):
        ax3.plot(state_probs[:, state], label=f'P(State {state})', 
                alpha=0.7, linewidth=0.5, color=colors[state])
    ax3.set_xlabel('Draw Number')
    ax3.set_ylabel('Probability')
    ax3.set_title('State Probabilities Over Time')
    ax3.legend()
    
    # --- Plot 4: BIC comparison ---
    ax4 = fig.add_subplot(5, 2, 4)
    states_list = sorted(results.keys())
    bic_vals = [results[s]['bic'] for s in states_list]
    aic_vals = [results[s]['aic'] for s in states_list]
    ax4.plot(states_list, bic_vals, 'bo-', label='BIC', markersize=8)
    ax4.plot(states_list, aic_vals, 'rs--', label='AIC', markersize=8)
    ax4.set_xlabel('Number of Hidden States')
    ax4.set_ylabel('Information Criterion')
    ax4.set_title('Model Selection: BIC & AIC')
    ax4.legend()
    ax4.set_xticks(states_list)
    
    # --- Plot 5: Number frequencies by regime ---
    ax5 = fig.add_subplot(5, 2, 5)
    width = 0.8 / n_states
    for state in range(n_states):
        mask = hidden_states == state
        state_draws = draws[mask].flatten()
        freq = np.bincount(state_draws, minlength=max_number + 1)[1:]
        freq_norm = freq / freq.sum()  # Normalize
        x_pos = np.arange(1, max_number + 1) + state * width - 0.4
        ax5.bar(x_pos, freq_norm, width=width, alpha=0.7, 
               color=colors[state], label=f'State {state}')
    ax5.axhline(y=1/max_number, color='black', linestyle='--', alpha=0.5, label='Expected (uniform)')
    ax5.set_xlabel('Lottery Number')
    ax5.set_ylabel('Relative Frequency')
    ax5.set_title('Number Frequency Distribution by Regime')
    ax5.legend()
    
    # --- Plot 6: Feature distributions by regime ---
    key_features = ['mean', 'std', 'sum_val', 'even_count']
    ax6 = fig.add_subplot(5, 2, 6)
    
    data_by_state = []
    labels = []
    for feat_idx, feat in enumerate(key_features):
        for state in range(n_states):
            mask = hidden_states == state
            data_by_state.append(features_df[feat][mask].values)
            labels.append(f'S{state}\n{feat}')
    
    positions = []
    pos = 0
    for feat_idx in range(len(key_features)):
        for state in range(n_states):
            positions.append(pos)
            pos += 1
        pos += 0.5  # Gap between features
    
    bp = ax6.boxplot(data_by_state, positions=positions, widths=0.6,
                     patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        state = i % n_states
        patch.set_facecolor(colors[state])
        patch.set_alpha(0.6)
    ax6.set_xticks(positions)
    ax6.set_xticklabels(labels, fontsize=7)
    ax6.set_title('Feature Distributions by Regime')
    
    # --- Plot 7: Draw sum time series colored by state ---
    ax7 = fig.add_subplot(5, 2, (7, 8))
    draw_sums = draws.sum(axis=1)
    for state in range(n_states):
        mask = hidden_states == state
        ax7.scatter(np.where(mask)[0], draw_sums[mask], 
                   c=[colors[state]], alpha=0.4, s=5, label=f'State {state}')
    ax7.set_xlabel('Draw Number')
    ax7.set_ylabel('Sum of Draw')
    ax7.set_title('Draw Sum Over Time (Colored by Detected State)')
    ax7.legend(markerscale=3)
    
    # --- Plot 8: Transition matrix heatmap ---
    ax8 = fig.add_subplot(5, 2, 9)
    from matplotlib.colors import LinearSegmentedColormap
    trans_mat = results[n_states]['model'].transmat_
    im = ax8.imshow(trans_mat, cmap='YlOrRd', vmin=0, vmax=1)
    for i in range(n_states):
        for j in range(n_states):
            ax8.text(j, i, f'{trans_mat[i,j]:.3f}', ha='center', va='center', fontsize=10)
    ax8.set_xlabel('To State')
    ax8.set_ylabel('From State')
    ax8.set_title('State Transition Matrix')
    ax8.set_xticks(range(n_states))
    ax8.set_yticks(range(n_states))
    plt.colorbar(im, ax=ax8)
    
    # --- Plot 9: Autocorrelation of draw sums ---
    ax9 = fig.add_subplot(5, 2, 10)
    max_lag = 20
    autocorrs = [1.0]  # lag 0
    for lag in range(1, max_lag + 1):
        corr = np.corrcoef(draw_sums[:-lag], draw_sums[lag:])[0, 1]
        autocorrs.append(corr)
    ax9.bar(range(max_lag + 1), autocorrs, alpha=0.7, color='steelblue')
    ax9.axhline(y=2/np.sqrt(len(draw_sums)), color='red', linestyle='--', alpha=0.5)
    ax9.axhline(y=-2/np.sqrt(len(draw_sums)), color='red', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Lag')
    ax9.set_ylabel('Autocorrelation')
    ax9.set_title('Autocorrelation of Draw Sums')
    
    plt.tight_layout()
    plt.savefig('/home/claude/hmm_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  ✓ Visualizations saved to hmm_analysis.png")


# =====================================================================
# STEP 7: PREDICTIVE ANALYSIS
# =====================================================================

def predict_next_regime(model, X, hidden_states):
    """
    Predict the probability of each regime for the NEXT draw,
    based on the current state.
    """
    print("\n" + "=" * 70)
    print("STEP 6: PREDICTION FOR NEXT DRAW")
    print("=" * 70)
    
    current_state = hidden_states[-1]
    trans_mat = model.transmat_
    
    print(f"\n  Current detected state: State {current_state}")
    print(f"\n  Probability of next draw being in each state:")
    for j in range(model.n_components):
        prob = trans_mat[current_state, j]
        print(f"    State {j}: {prob:.4f} ({prob*100:.1f}%)")
    
    # Multi-step forecast
    print(f"\n  5-step regime forecast:")
    state_dist = np.zeros(model.n_components)
    state_dist[current_state] = 1.0
    
    for step in range(1, 6):
        state_dist = state_dist @ trans_mat
        dominant = np.argmax(state_dist)
        probs_str = ", ".join([f"S{j}:{state_dist[j]:.3f}" for j in range(model.n_components)])
        print(f"    Draw +{step}: [{probs_str}] → Most likely: State {dominant}")


# =====================================================================
# STEP 8: ACCURACY CHECK (only when true states are known)
# =====================================================================

def check_accuracy(hidden_states, true_states):
    """Compare detected states with true states (for validation)."""
    print("\n" + "=" * 70)
    print("VALIDATION: ACCURACY CHECK (using known true states)")
    print("=" * 70)
    
    n_detected = len(np.unique(hidden_states))
    n_true = len(np.unique(true_states))
    
    if n_detected != n_true:
        print(f"\n  Note: HMM found {n_detected} states, true data has {n_true} states.")
        print(f"  Collapsing HMM states to best-match the 2 true regimes...")
        
        # For each HMM state, check which true state it aligns with most
        # Then map multiple HMM states to the best matching true state
        from itertools import product as iter_product
        
        best_acc = 0
        best_mapping = None
        # Try all possible mappings of detected states to {0, 1}
        for combo in iter_product(range(n_true), repeat=n_detected):
            mapped = np.array([combo[s] for s in hidden_states])
            acc = np.mean(mapped == true_states)
            if acc > best_acc:
                best_acc = acc
                best_mapping = combo
        
        mapped_states = np.array([best_mapping[s] for s in hidden_states])
        print(f"  Best mapping: {dict(enumerate(best_mapping))}")
        print(f"  Accuracy: {best_acc:.1%}")
    else:
        accuracy_direct = np.mean(hidden_states == true_states)
        accuracy_flipped = np.mean(hidden_states == (1 - true_states))
        best_acc = max(accuracy_direct, accuracy_flipped)
        mapped_states = hidden_states if accuracy_direct >= accuracy_flipped else (1 - hidden_states)
        print(f"\n  Accuracy: {best_acc:.1%}")
    
    from sklearn.metrics import classification_report
    print(f"\n  Classification Report:")
    print(classification_report(true_states, mapped_states,
                               target_names=['Fair', 'Rigged']))
    
    return best_acc


# =====================================================================
# MAIN: RUN THE FULL PIPELINE
# =====================================================================

def run_pipeline():
    """Execute the complete HMM analysis pipeline."""
    
    print("╔" + "═" * 68 + "╗")
    print("║" + "  HMM PIPELINE FOR RIGGED LOTTERY DETECTION  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    # --- Generate data (replace with your real data loading) ---
    # print("\n📊 STEP 1: GENERATING SYNTHETIC DATA")
    # print("   (Replace this with your real lottery data)")
    # draws, true_states, favored_numbers = generate_rigged_lottery(n_draws=1000)
    # print(f"   Generated {len(draws)} draws, 6 numbers each (1-49)")
    # print(f"   Hidden favored numbers: {favored_numbers}")
    
    # --- Load your real data ---
    print("\n📊 STEP 1: LOADING REAL LOTTERY DATA")
    df = pd.read_csv('Data.csv', sep=';')
    draws = df[['n1','n2','n3','n4','n5']].values
    true_states = None  # We don't know the true states
    print(f"   Loaded {len(draws)} draws")
    
    # --- Feature engineering ---
    print("\n📊 STEP 1b: ENGINEERING FEATURES")
    features_df = engineer_features(draws)
    print(f"   Extracted {len(features_df.columns)} features: {list(features_df.columns)}")
    
    # --- Randomness tests ---
    test_results = run_randomness_tests(draws)
    
    # --- Fit HMMs ---
    hmm_results, X, scaler = fit_hmm_models(features_df, max_states=6, n_fits=10)
    
    # --- Model selection ---
    best_n, best_model_info = select_best_model(hmm_results)
    
    # --- Regime analysis ---
    hidden_states, state_probs, regime_stats = analyze_regimes(
        best_model_info, X, features_df, draws, scaler
    )
    
    # --- Predictions ---
    predict_next_regime(best_model_info['model'], X, hidden_states)
    
    # --- Accuracy check (only with synthetic data) ---
    if true_states is not None:
        accuracy = check_accuracy(hidden_states, true_states)
    
    # --- Visualizations ---
    print("\n📊 CREATING VISUALIZATIONS...")
    create_visualizations(draws, features_df, hidden_states, state_probs,
                         hmm_results, true_states)
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  ┌──────────────────────────────────────────────────┐
  │  Draws analyzed:     {len(draws):>6}                      │
  │  Features extracted: {len(features_df.columns):>6}                      │
  │  Hidden states found:{best_n:>6}                      │
  │  Chi² p-value:       {test_results['chi2_p']:>10.6f}              │
  │  Entropy ratio:      {test_results['entropy_ratio']:>10.4f}              │
  └──────────────────────────────────────────────────┘
    """)
    
    if best_n > 1:
        print("  🔴 CONCLUSION: Evidence of multiple operating regimes detected.")
        print("     The lottery appears to have hidden states that differ in their")
        print("     statistical properties, consistent with a rigging mechanism.")
    else:
        print("  🟢 CONCLUSION: No strong evidence of multiple regimes.")
        print("     A single-state model best fits the data.")
    
    return draws, features_df, hidden_states, hmm_results, best_model_info


# Run it!
if __name__ == "__main__":
    results = run_pipeline()
