import os

from pathlib import Path
import os
import json
import random
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

import torch
from torch_geometric.data import Data

from functools import lru_cache
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split

# From local files
from data_extraction import get_file_list_from_github

# Suppress the future warning for this functionality
pd.set_option('future.no_silent_downcasting', True)

# File Paths
BASE_DIR = Path.cwd()
PARQUET_DIR = BASE_DIR / 'NBA_SportVU_Parquet'
CACHE_DIR = BASE_DIR / 'Dataset_Cache'
GRAPH_CACHE_DIR = BASE_DIR / 'Graph_Cache'
MODEL_DATA_DIR = BASE_DIR / 'Model_Data'

SPLIT_SAVE_PATH = MODEL_DATA_DIR / 'data_splits.json'
CHECKPOINT_PATH = MODEL_DATA_DIR / 'checkpoint_2.0.pt'
CHECKPOINT_LATE_PATH = MODEL_DATA_DIR / 'checkpoint_2.0_late.pt'
SHOOTING_DATA_PATH = BASE_DIR / 'shooting_data_filtered.csv'
MODEL_LATE_PATH = MODEL_DATA_DIR /'gnn_model_2.0_late.pt'


# Constant Globals
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0
WINDOW_SIZE = 25

MAX_FRAMES = 1500   # Discard possessions longer than 60 seconds (missing end event)

N_NODES = 12 # 10 players, the ball, and the basket
_src, _dst = zip(*[(i,j) for i in range(N_NODES) for j in range(N_NODES) if i != j])
EDGE_SRC = np.array(_src)
EDGE_DST = np.array(_dst)
EDGE_INDEX = torch.tensor([list(_src), list(_dst)], dtype=torch.long)

# Event Codes and Keywords
MADE_SHOT = 1
MISSED_SHOT = 2
FREE_THROW = 3
TURNOVER = 5
FOUL = 6
END_PERIOD = 13

POSSESSION_ENDING_EVENTS = {MADE_SHOT, MISSED_SHOT, TURNOVER, FOUL, END_PERIOD}

OFFENSIVE_FOUL_KEYWORDS = {'OFF.FOUL', 'OFFENSIVE CHARGE FOUL'}
NON_BONUS_FOUL_KEYWORDS = {'T.FOUL', 'FLAGRANT.FOUL.TYPE1'}
LOOSE_BALL_KEYWORD = 'L.B.FOUL'

MADE_KEYWORDS = {
        'DUNK', 'LAYUP', 'JUMP SHOT', 'HOOK SHOT', 'FINGER ROLL',
        'TIP SHOT', 'PUTBACK', 'DRIVING', 'CUTTING', 'PULLUP',
        'TURNAROUND', 'FADEAWAY', 'ALLEY OOP', 'BANK SHOT', 'FLOATER',
        'FLOATING', 'STEP BACK', 'RUNNING'
    }      



# =============================================
# Preprocessing Pipeline
# =============================================

# Stage 0: Load Game
@lru_cache(maxsize=4)
def load_game_cached(path: str):
    mom = pq.read_table(path, filters=[('record_type','=','moment')]).to_pandas()
    ball = pq.read_table(path, filters=[('record_type','=','ball')]).to_pandas()
    evt = pq.read_table(path, filters=[('record_type','=','event')]).to_pandas()

    for df in (mom, ball, evt):
        df['event_id'] = pd.to_numeric(df['event_id'], errors='coerce').astype('Int64')

    return mom, ball, evt

# Stage 1: Clean Up Missing Data
def resolve_possession_team(frames: pd.DataFrame) -> int:
    '''
    Determine the possession team for a possession's frames.

    Returns:
        team_id
        - team_id is None if possession team cannot be determined
    '''
    known = frames.loc[frames['possession_team_id'].notna(), 'possession_team_id']

    if known.empty:
        return None

    # mode() is O(n) and returns the most frequent value.
    # For a well-formed possession this should be unanimous,
    # but mode guards against any stray corrupt frames.
    team_id = int(known.mode().iloc[0])
    return team_id

def filter_live_frames(frames: pd.DataFrame, tol: float = 0.05) -> pd.DataFrame:
    if 'game_clock' not in frames.columns:
        return frames

    # Sort once, no extra copy
    frames = frames.sort_values('timestamp')

    gc = frames['game_clock'].to_numpy(dtype=np.float32, copy=False)

    # Build mask in NumPy (faster + no pandas overhead)
    valid = ~np.isnan(gc)

    # Compute diff manually (faster than pandas diff)
    delta = np.empty_like(gc)
    delta[0] = 0.0
    delta[1:] = gc[1:] - gc[:-1]

    decreasing = delta <= tol

    mask = valid & decreasing

    # Only copy once here
    return frames.loc[mask]

def fill_missing_tracking(frames: pd.DataFrame) -> pd.DataFrame:
    '''
    Fill missing values in tracking frames.

    Fill strategy by column:
      - home_team_id / visitor_team_id : forward-fill only (stable identifiers)
      - x, y, z                     : per-player linear interpolation, then
                                      ffill/bfill for leading/trailing gaps
      - possession_team_id          : filled externally by resolve_possession_team;
                                      not touched here
      - game_clock / shot_clock     : deduplicated to one value per timestamp,
                                      then linearly interpolated between bracketing
                                      known values with a min_gap >= 0.04 s check,
                                      then broadcast back to all rows at that timestamp.
                                      Timestamps where game_clock cannot be resolved
                                      are dropped entirely.
    '''
    frames = frames.sort_values(['player_id', 'timestamp']).copy()

    # Stable game identifiers: ffill only, no bfill
    # bfill would be wrong here — the home team at frame 0 is
    # always the home team; there's no valid 'previous' value to infer.
    for col in ('home_team_id', 'visitor_team_id'):
        if col in frames.columns:
            frames[col] = frames[col].ffill()

    # Tracking coordinates: interpolate per player/ball
    # Grouping by player_id naturally isolates the ball (player_id == -1)
    # as its own group, so z-interpolation works correctly for it too.
    # limit=None means we interpolate across arbitrarily long gaps;
    # tighten this (e.g. limit=5) if you want to reject large dropouts.
    for col in ('x', 'y', 'z'):
        if col in frames.columns:
            frames[col] = (
                frames.groupby('player_id', sort=False)[col]
                .transform(lambda s: s.interpolate(method='linear', limit_direction='both'))
            )
            # ffill/bfill handles leading or trailing NAs that interpolate can't reach
            frames[col] = frames[col].ffill().bfill()

    frames = frames.sort_values('timestamp').reset_index(drop=True)

    def interpolate_clock(series: pd.Series, timestamps: pd.Series, min_gap: float = 0.04) -> pd.Series:
        s = series.copy()
        nan_mask = s.isna().to_numpy()
        if not nan_mask.any():
            return s

        ts = timestamps.to_numpy(dtype=float)
        vals = s.to_numpy(dtype=float)

        known_mask = ~nan_mask
        known_ts = ts[known_mask]
        known_vals = vals[known_mask]

        if len(known_ts) < 2:
            return s  # Can't bracket anything

        # Interpolate all NaN positions in one shot
        interp_vals = np.interp(ts[nan_mask], known_ts, known_vals)

        # Rejection mask: find each NaN's bracketing known values and check clock span
        # searchsorted gives the insertion point — prev is one to the left, next is that point
        insert = np.searchsorted(known_ts, ts[nan_mask])
        has_both_sides = (insert > 0) & (insert < len(known_ts))

        prev_clock = np.where(has_both_sides, known_vals[np.clip(insert - 1, 0, len(known_vals) - 1)], np.nan)
        next_clock = np.where(has_both_sides, known_vals[np.clip(insert, 0, len(known_vals) - 1)], np.nan)
        clock_span = prev_clock - next_clock  # counts down, so positive when running

        valid = has_both_sides & (clock_span >= min_gap)
        interp_vals[~valid] = np.nan

        vals[nan_mask] = interp_vals

        # Coerce to avoid a FutureWarning from pandas
        if vals.dtype != np.float32:
            vals = vals.astype(np.float32)

        s[:] = vals
        return s


    if 'game_clock' in frames.columns or 'shot_clock' in frames.columns:
        # Get unique timestamps and their first clock values — one pass
        ts_unique, ts_idx = np.unique(frames['timestamp'].to_numpy(), return_index=True)

        gc_unique = frames['game_clock'].to_numpy()[ts_idx]
        sc_unique = frames['shot_clock'].to_numpy()[ts_idx]

        # Interpolate on the deduplicated series
        gc_series = interpolate_clock(pd.Series(gc_unique), pd.Series(ts_unique))
        sc_series = interpolate_clock(pd.Series(sc_unique), pd.Series(ts_unique))
        sc_series = sc_series.fillna(-1.0)

        # Build lookup arrays for fast broadcast — O(log n) per row via searchsorted
        gc_vals = gc_series.to_numpy(dtype=np.float32)
        sc_vals = sc_series.to_numpy(dtype=np.float32)

        row_ts = frames['timestamp'].to_numpy()
        row_idx = np.searchsorted(ts_unique, row_ts)

        frames['game_clock'] = gc_vals[row_idx]
        frames['shot_clock'] = sc_vals[row_idx]

        # Drop rows where game_clock couldn't be resolved
        frames = frames[~np.isnan(gc_vals[row_idx])]

    return frames

# Stage 2: Parse Possessions
def get_possessions(evt: pd.DataFrame) -> pd.DataFrame:
    evt = (
        evt.drop_duplicates('event_id')
        .sort_values(['quarter', 'game_clock'], ascending=[True, False])
        .reset_index(drop=True)
    )

    possessions = []
    start_idx = 0

    for i, row in evt.iterrows():
        if row['event_type'] in POSSESSION_ENDING_EVENTS:
            poss_rows = evt.iloc[
                start_idx:i+1]

            # Use mode of quarter across all rows in possession,
            # not just the ending event which may be null
            quarter_known = poss_rows['quarter'].dropna()
            quarter = quarter_known.mode().iloc[0] if not quarter_known.empty else pd.NA

            possessions.append({
                'start_event_id': poss_rows['event_id'].min(),
                'end_event_id':   poss_rows['event_id'].max(),
                # Use the ending event's possession_team_id — this is when
                # the data transitions to the new team, so it correctly
                # identifies who was on offense during this possession
                'possession_team_id': row['possession_team_id'],
                'quarter': quarter
            })
            start_idx = i + 1

    return pd.DataFrame(possessions).reset_index(names='possession_id')

# Stage 3: Label Possessions
def classify_foul_counts(desc_home: str, desc_away: str, possession_team_id: int,
                          home_team_id: int, visitor_team_id: int) -> bool:
    '''
    Returns True if this foul counts toward the defending team's bonus tally.

    Fouling team is inferred from which description column is populated.
    Loose ball fouls are only counted if the fouling team is the defending team.
    Offensive fouls, technicals, and flagrants are never counted.
    '''

    desc = ''
    fouling_team_id = None

    # Be explicit: only assign home if desc_home is a non-null, non-empty string
    # AND desc_away is not also populated (shouldn't happen but guards against it)
    has_home = isinstance(desc_home, str) and desc_home.strip() and desc_home.strip().lower() != 'nan'
    has_away = isinstance(desc_away, str) and desc_away.strip() and desc_away.strip().lower() != 'nan'

    if has_home and not has_away:
        desc = desc_home.upper()
        fouling_team_id = home_team_id
    elif has_away and not has_home:
        desc = desc_away.upper()
        fouling_team_id = visitor_team_id
    else:
        # Either both null (can't determine fouler) or both populated (ambiguous)
        return False

    if not desc or fouling_team_id is None:
        return False

    if any(kw in desc for kw in OFFENSIVE_FOUL_KEYWORDS | NON_BONUS_FOUL_KEYWORDS):
        return False

    if LOOSE_BALL_KEYWORD in desc:
        return fouling_team_id != possession_team_id

    # Safeguard: can't determine context without possession team
    try:
        if possession_team_id is None or np.isnan(float(possession_team_id)):
            return False
    except (TypeError, ValueError):
        return False

    return fouling_team_id != possession_team_id

def build_bonus_timeline(evt: pd.DataFrame) -> dict:
    '''
    Compute the event_id at which each (team_id, quarter) combination
    enters the bonus (5 qualifying fouls).

    Returns:
        dict mapping (team_id, quarter) -> event_id at which bonus becomes active,
        or None if the team never reached bonus in that quarter.
    '''
    # Now safe to deduplicate + sort
    evt = (
        evt.drop_duplicates(subset='event_id')
        .sort_values(['quarter', 'game_clock'], ascending=[True, False])
        .reset_index(drop=True)
    )

    home_team_id = evt['home_team_id'].iloc[0]
    visitor_team_id = evt['visitor_team_id'].iloc[0]

    # (team_id, quarter) -> event_id when bonus triggered, None until then
    bonus_active = {}
    foul_counts  = {}  # (team_id, quarter) -> int

    fouls = evt[evt['event_type'] == FOUL]

    for _, row in fouls.iterrows():
        quarter = row['quarter']
        possession_team_id = row['possession_team_id']

        desc_h = str(row.get('desc_home', '') or '')
        desc_a = str(row.get('desc_away', '') or '')
        counts = classify_foul_counts(
            desc_h, desc_a,
            possession_team_id, home_team_id, visitor_team_id
        )

        if not counts:
            continue

        has_home = isinstance(desc_h, str) and desc_h.strip() and desc_h.strip().lower() != 'nan'
        has_away = isinstance(desc_a, str) and desc_a.strip() and desc_a.strip().lower() != 'nan'

        if has_home and not has_away:
            fouling_team_id = home_team_id
        elif has_away and not has_home:
            fouling_team_id = visitor_team_id
        else:
            continue  # Can't determine fouling team, skip

        key = (fouling_team_id, quarter)
        foul_counts[key] = foul_counts.get(key, 0) + 1

        if foul_counts[key] == 5 and key not in bonus_active:
            bonus_active[key] = row['event_id']

    return bonus_active

def get_bonus_state(bonus_timeline: dict, defending_team_id: int,
                    quarter: int, event_id: int) -> bool:
    trigger_event_id = bonus_timeline.get((defending_team_id, quarter))
    if trigger_event_id is None:
        return False
    return event_id > trigger_event_id

def parse_points_from_desc(desc: str) -> int:
    if not isinstance(desc, str):
        return 0
    desc = desc.upper()

    if desc.startswith('MISS'):
        return 0

    # Free throw: always 1 point if made (no MISS prefix)
    if 'FREE THROW' in desc:
        return 1

    # 3-point shot
    if '3PT' in desc or '3-PT' in desc:
        return 3

    # Any other made shot is 2
    # Check for known made-shot keywords to avoid false positives on
    # non-scoring events that happen to share desc columns
    if any(kw in desc for kw in MADE_KEYWORDS):
        return 2

    return 0

def label_possessions(poss: pd.DataFrame, evt: pd.DataFrame) -> pd.DataFrame:
    evt_sorted = (
        evt.drop_duplicates('event_id')
        .sort_values(['quarter', 'game_clock'], ascending=[True, False])
        .reset_index(drop=True)
    )
    evt_lookup = evt_sorted.set_index('event_id')

    def get_points(row):
        end_row = evt_lookup.loc[row['end_event_id']]
        pts = parse_points_from_desc(end_row.get('desc_home', ''))
        if pts == 0:
            pts = parse_points_from_desc(end_row.get('desc_away', ''))

        # For foul-ending possessions, accumulate points from the immediately
        # following free throw sequence before the next non-FT event
        if end_row['event_type'] == FOUL:
            end_pos = evt_sorted.index[evt_sorted['event_id'] == row['end_event_id']][0]
            for _, ft_row in evt_sorted.iloc[end_pos + 1:].iterrows():
                if ft_row['event_type'] != FREE_THROW:
                    break
                # Score increment on the FT row means it was made
                score_delta = (
                    parse_points_from_desc(ft_row.get('desc_home', '')) +
                    parse_points_from_desc(ft_row.get('desc_away', ''))
                )
                pts += score_delta

        return float(pts)
    poss['points'] = poss.apply(get_points, axis=1)
    return poss

# Stage 4: Normalization
def normalize_to_offense(frames, possession_team_id, home_team_id, quarter):
    home_right = (quarter <= 2)
    is_home = possession_team_id == home_team_id
    attacks_right = (is_home == home_right)

    if not attacks_right:
        frames = frames.copy()
        frames['x'] = COURT_LENGTH - frames['x']
        frames['y'] = COURT_WIDTH - frames['y']

    return frames

# Stage 5: Smoothing
def smooth_tracking(frames):
    frames = frames.copy()
    for pid, grp in frames.groupby('player_id', sort=False):
        idx = grp.index
        frames.loc[idx, 'x'] = gaussian_filter1d(grp['x'], sigma=1)
        frames.loc[idx, 'y'] = gaussian_filter1d(grp['y'], sigma=1)
    return frames

# Stage 6: Build Windows
def build_window(frames, target_ts):
    timestamps = np.sort(frames['timestamp'].unique())
    idx = np.searchsorted(timestamps, target_ts)
    if idx >= len(timestamps) or timestamps[idx] != target_ts or idx < WINDOW_SIZE - 1:
        return None

    ts_start = timestamps[idx - WINDOW_SIZE + 1]
    ts_end = timestamps[idx]

    # Since frames is pre-sorted by timestamp, use direct boolean slice
    mask = (frames['timestamp'] >= ts_start) & (frames['timestamp'] <= ts_end)
    return frames[mask]

# Stage 7: Build Graphs
def load_player_stats(csv_path: str = SHOOTING_DATA_PATH) -> dict:
    '''
    Load player shooting stats from a CSV file and return a nested lookup map.

    Expected CSV columns: player_id, player_name, fg_pct, three_pt_pct, ft_pct

    Returns:
        dict mapping player_id (int) -> {stat_name (str) -> value (float)}
        e.g. {23: {'fg_pct': 0.502, 'three_pt_pct': 0.381, 'ft_pct': 0.774}, ...}
        Returns an empty dict if csv_path is None.
    '''
    if csv_path is None:
        return {}

    df = pd.read_csv(csv_path)
    stat_map = {}
    for row in df.itertuples(index=False):
        stat_map[int(row.player_id)] = {
            'two_pt_pct': float(row.two_pt_pct),
            'three_pt_pct': float(row.three_pt_pct),
            'ft_pct': float(row.ft_pct),
            'height': int(row.height)
        }
    return stat_map

def get_player_stats(stat_map: dict, player_id: int) -> tuple[float, float, float]:
    '''
    Look up shooting stats for a single player.

    Returns:
        (fg_pct, three_pt_pct, ft_pct) floats, defaulting to 0.0 if not found.
    '''
    entry = stat_map.get(int(player_id), {})
    return (
        entry.get('two_pt_pct', 0.0),
        entry.get('three_pt_pct', 0.0),
        entry.get('ft_pct', 0.0),
        entry.get('height', 0.0)
    )

def build_graph(window, possession_team_id, label, is_bonus: bool = False, stat_map: dict = None):
    T = WINDOW_SIZE
    # Channels: x, y, z, is_bonus, fg_pct, three_pt_pct, ft_pct, height = 8 total
    dyn_features = np.zeros((N_NODES, T, 3), dtype=np.float32)   # x, y, z
    static_features = np.zeros((N_NODES, 7), dtype=np.float32)   # is_bonus, 2pt, 3pt, ft, height

    timestamps = np.sort(window['timestamp'].unique())

    ball_df = window[window['record_type']=='ball']
    off_df = window[(window['record_type']=='moment') & (window['team_id']==possession_team_id)].copy()
    def_df = window[(window['record_type']=='moment') & (window['team_id']!=possession_team_id)].copy()

    off_df['node_i'] = off_df.groupby('timestamp').cumcount().clip(upper=4)
    def_df['node_i'] = def_df.groupby('timestamp').cumcount().clip(upper=4)

    off_df = off_df[off_df['node_i'] < 5]
    def_df = def_df[def_df['node_i'] < 5]

    # Offensive players
    off_idx = off_df['node_i'].to_numpy()
    off_t = np.searchsorted(timestamps, off_df['timestamp'].to_numpy())
    off_xy = off_df[['x','y']].to_numpy(dtype=np.float32)

    dyn_features[off_idx, off_t, 0:2] = off_xy

    # Defensive players
    def_idx = def_df['node_i'].to_numpy() + 5
    def_t = np.searchsorted(timestamps, def_df['timestamp'].to_numpy())
    def_xy = def_df[['x','y']].to_numpy(dtype=np.float32)

    dyn_features[def_idx, def_t, 0:2] = def_xy

    # Ball
    ball_t = np.searchsorted(timestamps, ball_df['timestamp'].to_numpy())
    ball_xyz = ball_df[['x','y','z']].to_numpy(dtype=np.float32)

    dyn_features[10, ball_t, 0:3] = ball_xyz

    # Global, Graph Level Features
    shot_clock_val = window['shot_clock'].iloc[-1]
    game_clock_val = window['game_clock'].iloc[-1]

    if np.isnan(game_clock_val):
        game_clock_val = -1.0

    if np.isnan(shot_clock_val):
        if 0 <= game_clock_val<= 24.0:
            shot_clock_val = game_clock_val
        else:
            shot_clock_val = -1.0

    if game_clock_val == -1.0 or shot_clock_val == -1.0:
        return None

    u = torch.tensor([[
        shot_clock_val / 24.0,   # normalize to [0, 1]
        game_clock_val / 720.0   # normalize to [0, 1] (12 min quarters)
    ]], dtype=torch.float)

    # Basket (node 11) — fixed position, broadcast across all timesteps
    # After normalize_to_offense, offense always attacks the right basket at (89, 25)
    BASKET_X, BASKET_Y = 89.0, 25.0
    static_features[11, 0] = BASKET_X
    static_features[11, 1] = BASKET_Y

    if is_bonus:
            static_features[:5, 2] = 1.0

    if stat_map:
        for df_group, idx_offset in ((off_df, 0), (def_df, 5)):
            for node_i, grp in df_group.groupby('node_i'):
                player_id = grp['player_id'].iloc[0]
                two_pt, three_pt, ft, height = get_player_stats(stat_map, player_id)
                node_idx = node_i + idx_offset
                # Stats are static per player so broadcast across all T timesteps
                static_features[node_idx, 3] = two_pt
                static_features[node_idx, 4] = three_pt
                static_features[node_idx, 5] = ft
                static_features[node_idx, 6] = (height - 50.0) / 100.0 # Rough normalization for height

    x_dyn = torch.tensor(dyn_features.reshape(N_NODES, -1), dtype=torch.float)
    x_static = torch.tensor(static_features, dtype=torch.float)

    last_pos = dyn_features[:, -1, :2]
    diff = last_pos[EDGE_SRC] - last_pos[EDGE_DST]
    edge_attr = torch.tensor(np.linalg.norm(diff, axis=1, keepdims=True), dtype=torch.float)

    return Data(
        x=x_dyn,
        x_static=x_static,
        edge_index=EDGE_INDEX,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.float),
        u=u
    )



# Execute preprocessing (up to but not including build_windows)
def build_index(PARQUET_DIR, files):
    '''
    Driver for creation of the possession by possession metadata and frames parquet files
    '''
    sample_id = 0
    skipped_no_team = 0
    skipped_short = 0
    skipped_long = 0
    skipped_live_filtering = 0
    skipped_clock = 0

    for f in tqdm(files, desc='Building index'):
        index_rows = []
        frames_rows = []

        path = os.path.join(PARQUET_DIR, f['stem'] + '.parquet')
        if not os.path.exists(path):
            continue

        frames_path = os.path.join(CACHE_DIR, f'{f['stem']}_frames.parquet')
        metadata_path = os.path.join(CACHE_DIR, f'{f['stem']}_metadata.parquet')

        # Skip games already fully indexed
        if os.path.exists(metadata_path) and os.path.exists(frames_path):
            existing = pd.read_parquet(metadata_path)
            sample_id = int(existing['sample_id'].max()) + 1
            print(f'\nSkipping {f['stem']} — already indexed ({len(existing)} samples).')
            continue

        mom, ball, evt = load_game_cached(path)

        # Initial robust checks for evt before any modifications
        if evt.empty:
            print(f'\nSkipping game {f['stem']} due to empty event data.')
            continue
        if 'quarter' not in evt.columns or 'game_clock' not in evt.columns:
            continue

        evt_cleaned = evt.dropna(subset=['quarter', 'game_clock']).copy()
        if evt_cleaned.empty:
            continue


        # Ensure quarter is integer type for consistency with other parts of the code
        evt_cleaned['quarter'] = evt_cleaned['quarter'].astype('int64', errors='ignore')

        home_team_id = evt_cleaned['home_team_id'].iloc[0]
        visitor_team_id = evt_cleaned['visitor_team_id'].iloc[0]
        bonus_timeline = build_bonus_timeline(evt_cleaned)

        possessions = get_possessions(evt_cleaned)
        possessions = label_possessions(possessions, evt_cleaned)

        for poss in possessions.itertuples(index=False):

            poss_evt = evt_cleaned[evt_cleaned['event_id'].between(
                poss.start_event_id, poss.end_event_id
            )]

            poss_mom = mom[mom['event_id'].between(
                poss.start_event_id, poss.end_event_id
            )].copy()

            poss_ball = ball[ball['event_id'].between(
                poss.start_event_id, poss.end_event_id
            )].copy()

            poss_mom['record_type'] = 'moment'
            poss_ball['record_type'] = 'ball'
            poss_ball['team_id'] = -1
            poss_ball['player_id'] = -1

            poss_mom = poss_mom.drop(columns=['possession_team_id'], errors='ignore')
            poss_ball = poss_ball.drop(columns=['possession_team_id'], errors='ignore')

            frames = pd.concat([poss_mom, poss_ball], ignore_index=True)
            frames['timestamp'] = frames['timestamp'].astype(np.int64)

            # Join possession_team_id from evt by event_id
            # Note: Using evt_cleaned for merge to ensure consistency with bonus_timeline and possessions
            evt_team = poss_evt[['event_id', 'possession_team_id']].drop_duplicates('event_id')
            frames = frames.merge(evt_team, on='event_id', how='left')

            # This merge might add new rows or modify existing ones; ensure 'quarter' is still present for filtering
            # It's safer to re-check or ensure quarter is propagated correctly if it's needed in frames directly
            # The original logic uses frames['quarter'] later for mode() after filter_live_frames
            frames = filter_live_frames(frames)

            if len(frames) == 0:
                skipped_live_filtering += 1 # Count this as live filtering, as it's a frames-level issue
                continue

            # Now extract quarter from frames, which might have been altered by merges/filters
            valid_quarters = frames['quarter'].dropna()

            if valid_quarters.empty:
                skipped_live_filtering += 1 # Count this as live filtering, as it's a frames-level issue
                continue

            # Use mode to get the quarter, as a single quarter should dominate a possession
            quarter = int(valid_quarters.mode().iloc[0])

            # Resolve possession team (mirrors process_possession_cached)
            resolved_team = resolve_possession_team(frames)
            if resolved_team is None:
                # print(f'Skipped possession in game {f['stem']} (start_event_id: {poss.start_event_id}) due to unresolved possession team.')
                skipped_no_team += 1
                continue

            defending_team_id = (
                visitor_team_id if resolved_team == home_team_id else home_team_id
            )
            is_bonus = get_bonus_state(bonus_timeline, defending_team_id, quarter, poss.start_event_id)

            frames = normalize_to_offense(frames, resolved_team, home_team_id, quarter)
            frames = fill_missing_tracking(frames)
            frames = smooth_tracking(frames)

            timestamps = np.sort(frames['timestamp'].unique())
            n_frames = len(timestamps)

            if n_frames < WINDOW_SIZE:
                skipped_short += 1
                continue

            if n_frames > MAX_FRAMES:
                skipped_long += 1
                continue

            # Build a per-timestamp clock lookup once before the loop
            clock_by_ts = frames.groupby('timestamp')[['game_clock', 'shot_clock']].first()

            frames_saved = False

            for i in range(WINDOW_SIZE - 1, len(timestamps)):
                target_ts = timestamps[i]

                game_clock_val = clock_by_ts.at[target_ts, 'game_clock'] if target_ts in clock_by_ts.index else np.nan
                shot_clock_val = clock_by_ts.at[target_ts, 'shot_clock'] if target_ts in clock_by_ts.index else np.nan

                if np.isnan(game_clock_val):
                    skipped_clock += 1
                    continue

                if np.isnan(shot_clock_val):
                    if 0 <= game_clock_val <= 24.0:
                        shot_clock_val = game_clock_val
                    else:
                        skipped_clock += 1
                        continue

                if not frames_saved:
                    # Save the frames
                    frames['start_event_id'] = poss.start_event_id  # important so __getitem__ can slice
                    frames['end_event_id'] = poss.end_event_id
                    frames_rows.append(frames)
                    frames_saved = True

                index_rows.append({
                    'sample_id': sample_id,
                    'game_path': path,
                    'frames_path': frames_path,
                    'start_event_id': poss.start_event_id,
                    'end_event_id': poss.end_event_id,
                    'timestamp': target_ts,
                    'possession_team_id': resolved_team,
                    'points': poss.points,
                    'quarter': quarter,
                    'is_bonus': is_bonus
                })
                sample_id += 1
        if index_rows:
            game_index_df = pd.DataFrame(index_rows)
            game_frames_df = pd.concat(frames_rows, ignore_index=True) if frames_rows else pd.DataFrame()

            game_index_df.to_parquet(metadata_path)
            if not game_frames_df.empty:
                game_frames_df.to_parquet(frames_path)

    print('Possessions across all games:')
    print(f'Skipped (no possession team): {skipped_no_team}')
    print(f'Skipped (too short): {skipped_short}')
    print(f'Skipped (too long): {skipped_long}')
    print(f'Skipped (stopped clock or invalid quarter in frames): {skipped_live_filtering}')
    print(f'Skipped (missing clock): {skipped_clock}')

# Driver to create the index - Note: resource intensive process
def build_index_for_files(files):
    '''
    Wrapper to build_index resumable 
    '''
    sample_files = sorted(
        f['stem'] for f in files
        if os.path.exists(os.path.join(PARQUET_DIR, f'{f['stem']}.parquet'))
    )

    # Convert list of stems to the dict format build_index expects
    files_for_index = [{'stem': stem} for stem in sample_files]

    # Make this resumable
    missing_files = []
    processed_files = 0
    for f in files_for_index:
        if os.path.exists(os.path.join(CACHE_DIR, f'{f['stem']}_metadata.parquet')):
            processed_files += 1
            continue

        missing_files.append(f)

    print(f'Found {processed_files} already in index.')

    # Output files to directory
    build_index(
        PARQUET_DIR=PARQUET_DIR,
        files=missing_files,
    )


# Build graph cache file for metadata file
def ensure_graph_exists(meta_path):
    '''
    Add graphs file to graph cache if it doesn't exist yet
    '''
    stem = Path(meta_path).stem.replace('_metadata', '')
    graph_path = Path(GRAPH_CACHE_DIR) / f'{stem}_graphs.pt'
    tmp_path   = Path(GRAPH_CACHE_DIR) / f'{stem}_graphs.tmp.pt'

    if graph_path.exists():
        return

    print(f'Building: {stem}')

    df = pd.read_parquet(meta_path)
    graphs = {}
    frames_cache = {}

    stat_map = load_player_stats()

    for (frames_key, start_eid, end_eid), poss_df in df.groupby(
        ['frames_path', 'start_event_id', 'end_event_id']
    ):
        if frames_key not in frames_cache:
            df_frames = pd.read_parquet(frames_key)
            frames_cache[frames_key] = {
                key: grp for key, grp in df_frames.groupby(['start_event_id', 'end_event_id'])
            }

        poss_frames = frames_cache[frames_key].get((start_eid, end_eid))
        if poss_frames is None:
            continue

        for i, (_, row) in enumerate(poss_df.iterrows()):
            if i % 5 != 0:
                continue

            window = build_window(poss_frames, row['timestamp'])
            if window is None:
                continue

            graph = build_graph(
                window,
                row['possession_team_id'],
                row['points'],
                row['is_bonus'],
                stat_map
            )

            if graph is None:
                continue

            graphs[int(row['sample_id'])] = graph

        # periodic save (safe resume)
        if len(graphs) % 5000 == 0:
            torch.save(graphs, tmp_path)

    frames_cache.clear()
    torch.save(graphs, graph_path)

    if tmp_path.exists():
        tmp_path.unlink()

    print(f'  Saved {len(graphs)} graphs')

# Driver to generate graph_cache for cleaned train/test set  
def build_good_graph_cache():
    '''
    Ensure we can build graphs for all of the files in the train/test set
    Save graphs to Graph_Cache
    Save file split to Model_Data
    '''
    bad_games = {
        '01.22.2016.IND.at.GSW',
        '11.02.2015.PHX.at.LAC',
        '11.04.2015.PHI.at.MIL'
    }

    # Load all files
    all_index_files = sorted([
        os.path.join(CACHE_DIR, f)
        for f in os.listdir(CACHE_DIR)
        if f.endswith('_metadata.parquet')
    ])

    bad_meta_files = {
        os.path.join(CACHE_DIR, f'{g}_metadata.parquet')
        for g in bad_games
    }

    # Initial Split
    train_files, temp_files = train_test_split(
        all_index_files,
        train_size=50,
        random_state=7
    )

    val_files, test_files = train_test_split(
        temp_files,
        train_size=5,
        test_size=5,
        random_state=7
    )

    # Remove bad files from train
    clean_train = [f for f in train_files if f not in bad_meta_files]
    num_removed = len(train_files) - len(clean_train)

    # Sample replacements from unused files
    used_files = set(train_files + val_files + test_files)

    unused = [
        f for f in all_index_files
        if f not in used_files
        and f not in bad_meta_files
    ]

    random.seed(7)
    replacements = random.sample(unused, num_removed)

    # Delete only bad graph files
    deleted = 0
    for bad in bad_meta_files:
        stem = Path(bad).stem.replace('_metadata', '')
        graph_path = Path(GRAPH_CACHE_DIR) / f'{stem}_graphs.pt'

        if graph_path.exists():
            graph_path.unlink()
            deleted += 1

    # Final train set
    final_train = clean_train + replacements

    # Build only what is needed
    needed_files = set(final_train + val_files + test_files)

    for f in needed_files:
        ensure_graph_exists(f)

    # Save splits
    splits = {
        'train': final_train,
        'val':   val_files,
        'test':  test_files,
    }

    with open(SPLIT_SAVE_PATH, 'w') as f:
        json.dump(splits, f, indent=2)

    print('\nFinal split:')
    print(f'  train: {len(final_train)}')
    print(f'  val:   {len(val_files)}')
    print(f'  test:  {len(test_files)}')


if __name__ == '__main__':
    for d in [PARQUET_DIR, CACHE_DIR, GRAPH_CACHE_DIR, MODEL_DATA_DIR]:
        os.makedirs(d, exist_ok=True)

    files = get_file_list_from_github()
    build_index_for_files(files)
    build_good_graph_cache()