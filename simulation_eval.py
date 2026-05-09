from pathlib import Path
import os
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# File Paths
BASE_DIR = Path.cwd()
PARQUET_DIR = BASE_DIR / 'NBA_SportVU_Parquet'
CACHE_DIR = BASE_DIR / 'Dataset_Cache'
GRAPH_CACHE_DIR = BASE_DIR / 'Graph_Cache'
MODEL_DATA_DIR = BASE_DIR / 'Model_Data'
RESULTS_DIR = BASE_DIR / 'Results'

# Make all directories
for d in [PARQUET_DIR, CACHE_DIR, GRAPH_CACHE_DIR, MODEL_DATA_DIR]:
    os.makedirs(d, exist_ok=True)

GBC_PASS_PATH   = MODEL_DATA_DIR / 'gbc_pass_model_real_only.joblib'
GNN_EPV_PATH    = MODEL_DATA_DIR / 'gnn_model_2.0_late.pt'

PASSES_CSV = BASE_DIR / 'all_games_passes_3.0.csv'
INDEX_FILE = CACHE_DIR / '11.18.2015.DAL.at.BOS_metadata.parquet'

GAME_STEM = INDEX_FILE.stem.replace('_metadata', '')
SIMULATION_RESULTS_PATH = RESULTS_DIR / f'{GAME_STEM}_pass_epv.csv'


if __name__ == '__main__':
    simulation_df = pd.read_csv(SIMULATION_RESULTS_PATH)
    group_keys = ['start_event_id', 'end_event_id', 'clock_norm', 'handler_id']

    # Per decision: actual vs best available (both metrics)
    decisions = simulation_df.groupby(group_keys)

    def summarize_decision(g):
        actual_rows  = g[g['is_actual_receiver']]
        optimal_rows = g[g['is_optimal_receiver']]

        if len(actual_rows) == 0 or len(optimal_rows) == 0:
            return None

        highest_future_receiver = g.loc[g['candidate_future_epv'].idxmax(), 'candidate_receiver_id']
        actual_receiver_id      = actual_rows['candidate_receiver_id'].values[0]

        return pd.Series({
            'actual_expected_epv':     actual_rows['candidate_expected_epv'].values[0],
            'actual_future_epv':       actual_rows['candidate_future_epv'].values[0],
            'optimal_expected_epv':    optimal_rows['candidate_expected_epv'].values[0],
            'highest_future_epv':      g['candidate_future_epv'].max(),
            'threw_to_optimal':        actual_rows['is_optimal_receiver'].values[0],
            'threw_to_highest_future': actual_receiver_id == highest_future_receiver,
        })

    decision_summary = (
        simulation_df
        .groupby(group_keys)
        .apply(summarize_decision)
        .dropna()
        .reset_index()
    )

    # Quick diagnostic on what was dropped
    total_groups = simulation_df.groupby(group_keys).ngroups
    print(f'Total decisions : {total_groups}')
    print(f'Clean decisions : {len(decision_summary)}')
    print(f'Dropped         : {total_groups - len(decision_summary)}')

    # Gap analysis
    decision_summary['expected_epv_gap'] = (
        decision_summary['optimal_expected_epv'] - decision_summary['actual_expected_epv']
    )
    decision_summary['future_epv_gap'] = (
        decision_summary['highest_future_epv'] - decision_summary['actual_future_epv']
    )

    print('=== How Closely Actual Pass Matches Best Available ===')
    print(f'  Risk-adjusted (expected EPV)')
    print(f'    Mean gap        : {decision_summary['expected_epv_gap'].mean():.4f}')
    print(f'    Median gap      : {decision_summary['expected_epv_gap'].median():.4f}')
    print(f'    % exact match   : {decision_summary['threw_to_optimal'].mean():.1%}')
    print(f'    % within 0.05   : {(decision_summary['expected_epv_gap'] <= 0.05).mean():.1%}')
    print(f'    % within 0.10   : {(decision_summary['expected_epv_gap'] <= 0.10).mean():.1%}')

    print(f'\n  Raw position (future EPV)')
    print(f'    Mean gap        : {decision_summary['future_epv_gap'].mean():.4f}')
    print(f'    Median gap      : {decision_summary['future_epv_gap'].median():.4f}')
    print(f'    % exact match   : {decision_summary['threw_to_highest_future'].mean():.1%}')
    print(f'    % within 0.05   : {(decision_summary['future_epv_gap'] <= 0.05).mean():.1%}')
    print(f'    % within 0.10   : {(decision_summary['future_epv_gap'] <= 0.10).mean():.1%}')

    # Correlation: do actual passes track expected or future EPV more closely?
    r_expected, _ = pearsonr(decision_summary['actual_expected_epv'], decision_summary['optimal_expected_epv'])
    r_future,   _ = pearsonr(decision_summary['actual_future_epv'],   decision_summary['highest_future_epv'])
    rho_expected, _ = spearmanr(decision_summary['actual_expected_epv'], decision_summary['optimal_expected_epv'])
    rho_future,   _ = spearmanr(decision_summary['actual_future_epv'],   decision_summary['highest_future_epv'])

    print(f'\n=== Correlation: Actual vs Best Available ===')
    print(f'  Expected EPV  — Pearson: {r_expected:.3f}  Spearman: {rho_expected:.3f}')
    print(f'  Future EPV    — Pearson: {r_future:.3f}  Spearman: {rho_future:.3f}')


    # ============================================================
    # FIGURE 1 — Ranked Expected EPV by Game
    # ============================================================
    csv_paths = sorted(glob.glob(f'{RESULTS_DIR}/*.csv'))
    if not csv_paths:
        raise FileNotFoundError(f'No CSVs found in {RESULTS_DIR}')


    # ============================================================
    # FIGURE 3 — Ranked Expected EPV by Game (5 bars)
    # ============================================================

    def compute_ranked_epvs(df):
        group_keys = ['start_event_id', 'end_event_id', 'clock_norm', 'handler_id']

        rank1, rank2, rank3, rank4, actual = [], [], [], [], []

        for _, g in df.groupby(group_keys):
            actual_rows = g[g['is_actual_receiver']]
            if len(actual_rows) == 0:
                continue

            ranked = g.sort_values('candidate_expected_epv', ascending=False).reset_index(drop=True)

            rank1.append(ranked.loc[0, 'candidate_expected_epv'])
            rank2.append(ranked.loc[1, 'candidate_expected_epv'] if len(ranked) > 1 else np.nan)
            rank3.append(ranked.loc[2, 'candidate_expected_epv'] if len(ranked) > 2 else np.nan)
            rank4.append(ranked.loc[3, 'candidate_expected_epv'] if len(ranked) > 3 else np.nan)
            actual.append(actual_rows['candidate_expected_epv'].values[0])

        return {
            'rank1':  np.nanmean(rank1),
            'rank2':  np.nanmean(rank2),
            'rank3':  np.nanmean(rank3),
            'rank4':  np.nanmean(rank4),
            'actual': np.nanmean(actual),
            'n':      len(rank1),
        }

    # Load all result files 
    game_stats = {}
    for path in csv_paths:
        stem = Path(path).stem.replace('_pass_epv', '')
        df   = pd.read_csv(path)
        game_stats[stem] = compute_ranked_epvs(df)
        print(f'  {stem}: {game_stats[stem]['n']} decisions')

    # Build plot data 
    games       = list(game_stats.keys())
    rank1_vals  = [game_stats[g]['rank1']  for g in games]
    rank2_vals  = [game_stats[g]['rank2']  for g in games]
    rank3_vals  = [game_stats[g]['rank3']  for g in games]
    rank4_vals  = [game_stats[g]['rank4']  for g in games]
    actual_vals = [game_stats[g]['actual'] for g in games]
    n_vals      = [game_stats[g]['n']      for g in games]

    x      = np.arange(len(games))
    width  = 0.16
    offset = [-2, -1, 0, 1, 2]

    colors = ['#1565C0', '#2196F3', '#64B5F6', '#BBDEFB', '#F44336']
    labels = ['Best (Rank 1)', '2nd Best', '3rd Best', '4th Best', 'Actual Choice']

    fig, ax = plt.subplots(figsize=(max(9, len(games) * 2.6), 6))

    bars = []
    for i, (vals, color, label) in enumerate(zip(
        [rank1_vals, rank2_vals, rank3_vals, rank4_vals, actual_vals], colors, labels
    )):
        b = ax.bar(x + offset[i] * width, vals, width,
                label=label, color=color, edgecolor='white', linewidth=0.6, zorder=3)
        bars.append(b)

    for xi, n in zip(x, n_vals):
        ax.text(xi, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.1,
                f'n={n}', ha='center', va='bottom', fontsize=8, color='#555')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [g.replace('.', ' ').replace('_', '\n') for g in games],
        fontsize=9, rotation=15, ha='right'
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax.set_ylabel('Mean Expected EPV  (pass_prob × future_EPV)', fontsize=11)
    ax.set_title('Pass Decision Quality by Game\nRanked Expected EPV vs Actual Choice',
                fontsize=13, fontweight='bold', pad=14)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    for bar_group in bars:
        for bar in bar_group:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=7, color='#333')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/fig3_ranked_epv_by_game.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved → fig3_ranked_epv_by_game.png')

    def compute_lowest_prob_selection_rate(df):
        group_keys = ['start_event_id', 'end_event_id', 'clock_norm', 'handler_id']

        lowest_count = 0
        total = 0

        lowest_future_epvs_all = []
        lowest_future_epvs_chosen = []
        chosen_probs_when_lowest = []

        for _, g in df.groupby(group_keys):
            actual_rows = g[g['is_actual_receiver']]
            if len(actual_rows) == 0:
                continue

            # Identify lowest-probability pass(es)
            min_prob = g['candidate_pass_prob'].min()
            lowest_rows = g[np.isclose(g['candidate_pass_prob'], min_prob)]

            # Average future EPV for lowest options (handles ties)
            lowest_future_epv = lowest_rows['candidate_future_epv'].mean()
            lowest_future_epvs_all.append(lowest_future_epv)

            # Actual chosen pass
            actual_prob = actual_rows['candidate_pass_prob'].values[0]

            if np.isclose(actual_prob, min_prob):
                lowest_count += 1

                lowest_future_epvs_chosen.append(
                    actual_rows['candidate_future_epv'].values[0]
                )

                chosen_probs_when_lowest.append(actual_prob)

            total += 1

        rate = lowest_count / total if total > 0 else np.nan

        return {
            'lowest_count': lowest_count,
            'total': total,
            'rate': rate,
            'avg_lowest_future_epv_chosen': np.nanmean(lowest_future_epvs_chosen),
            'avg_prob_when_lowest_chosen': np.nanmean(chosen_probs_when_lowest)
        }


    lowest_prob_stats = {}

    for path in csv_paths:
        stem = Path(path).stem.replace('_pass_epv', '')
        df = pd.read_csv(path)

        stats = compute_lowest_prob_selection_rate(df)
        lowest_prob_stats[stem] = stats

        print(f'{stem}:')
        print(f'  lowest chosen: {stats['lowest_count']} / {stats['total']}')
        print(f'  rate: {stats['rate']:.4f}')
        print(f'  avg lowest future EPV (chosen): {stats['avg_lowest_future_epv_chosen']:.4f}')
        print(f'  avg prob when lowest chosen:    {stats['avg_prob_when_lowest_chosen']:.4f}')


    csv_paths = sorted(glob.glob(f'{RESULTS_DIR}/*.csv'))

    all_results = []

    group_keys = ['start_event_id', 'end_event_id', 'clock_norm', 'handler_id']

    for path in csv_paths:
        df = pd.read_csv(path)

        def summarize_decision(g):
            actual_rows = g[g['is_actual_receiver']]
            if len(actual_rows) == 0:
                return None

            actual_future_epv = actual_rows['candidate_future_epv'].values[0]
            baseline_epv      = actual_rows['baseline_epv'].values[0]

            return pd.Series({
                'actual_future_epv': actual_future_epv,
                'baseline_epv': baseline_epv,
                'actual_beats_baseline': actual_future_epv > baseline_epv
            })

        decision_summary = (
            df.groupby(group_keys)
            .apply(summarize_decision)
            .dropna()
            .reset_index()
        )

        total = len(decision_summary)
        wins  = decision_summary['actual_beats_baseline'].sum()

        all_results.append({
            'game': Path(path).stem,
            'total_decisions': total,
            'beats_baseline': wins,
            'rate': wins / total if total > 0 else 0.0
        })

    # Combine all games 
    results_df = pd.DataFrame(all_results)

    overall_total = results_df['total_decisions'].sum()
    overall_wins  = results_df['beats_baseline'].sum()
    overall_rate  = overall_wins / overall_total

    print('=== Per Game ===')
    print(results_df.sort_values('rate', ascending=False))

    print('\n=== Overall ===')
    print(f'Total decisions        : {overall_total}')
    print(f'Actual > baseline      : {overall_wins}')
    print(f'Rate                   : {overall_rate:.2%}')
