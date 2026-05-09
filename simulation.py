import joblib
from pathlib import Path
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from collections import Counter

from scipy.stats import pearsonr, spearmanr
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from preprocessing import (
    load_player_stats, build_window, build_graph,
)
from utils.pass_features import (
    distance, compute_player_velocity, check_pass_success,
    compute_pass_angle, receiver_defender_closing_speed, pass_trajectory_crowding,
    nearest_defender_to_trajectory, passer_nearest_defender_dist, receiver_nearest_defender_dist,
    receiver_separation_ratio, offensive_spacing, max_defender_lane_depth
)
from pass_success_model_train import detect_passes, dataframe_to_frames

from epv_model_train_eval import TemporalGNN



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


# ============================================================
# GLOBAL CONSTANTS
# ============================================================
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0
WINDOW_SIZE = 25
N_NODES = 12

_src, _dst = zip(*[(i,j) for i in range(N_NODES) for j in range(N_NODES) if i != j])
EDGE_SRC = np.array(_src)
EDGE_DST = np.array(_dst)
EDGE_INDEX = torch.tensor([list(_src), list(_dst)], dtype=torch.long)

POSSESSION_RADIUS = 5.0
MIN_PASS_DISTANCE = 2.0
MIN_BALL_SPEED = 3.0
MIN_POSSESSION_FRAMES = 3
SUCCESS_POSSESSION_FRAMES = 5

PASS_FEATURE_COLS = [
    'pass_distance', 'pass_speed', 'pass_angle',
    'nearest_defender_dist', 'max_defender_lane_depth',
    'pass_trajectory_crowding', 'passer_velocity',
    'passer_nearest_defender_dist', 'receiver_velocity',
    'receiver_nearest_defender_dist', 'receiver_defender_closing_speed',
    'receiver_separation_ratio', 'offensive_spacing',
]

QUARTER_DURATION  = 12 * 60
OVERTIME_DURATION =  5 * 60

# ============================================================
# SIMULATION CONFIGURATION
# ============================================================
@dataclass
class SimConfig:
    z_carry_threshold:    float = 8.0
    min_handler_frames:   int   = 5
    min_stint_frames:     int   = 10
    sample_every_n:       int   = 5
    pass_delta_t_frames:  int   = 13
    velocity_k_frames:    int   = 3
    z_hand_height:        float = 5.5
    g_ft_per_s2:          float = 32.174
    fps:                  float = 25.0
    handler_switch_window: int  = 25

CFG = SimConfig()

_stat_map = None
def get_stat_map():
    global _stat_map
    if _stat_map is None:
        _stat_map = load_player_stats()
    return _stat_map

# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class FrameResult:
    timestamp:               int
    handler_id:              int
    baseline_epv:            float
    receiver_epvs:           dict
    receiver_pass_probs:     dict
    best_receiver_id:        int
    best_delta_epv:          float
    actual_next_handler:     Optional[int]  = None
    actual_pass_was_optimal: Optional[bool] = None

@dataclass
class DetectedPass:
    timestamp:            int
    from_player:          int
    to_player:            int
    start_pos:            tuple
    end_pos:              tuple
    pass_distance:        float
    pass_speed:           float
    success:              int
    aligned_with_optimal: Optional[bool] = None
    optimal_receiver:     Optional[int]  = None

@dataclass
class PossessionSummary:
    possession_id:         int
    game_path:             str
    possession_team_id:    int
    actual_points:         float
    frame_results:         list = field(default_factory=list)
    detected_passes:       list = field(default_factory=list)
    n_frames_evaluated:    int   = 0
    n_passes_observed:     int   = 0
    n_optimal_passes:      int   = 0
    mean_best_delta_epv:   float = 0.0
    optimal_pass_rate:     float = 0.0
    n_detected_passes:     int   = 0
    n_aligned_passes:      int   = 0
    pass_alignment_rate:   float = 0.0
    mean_pass_success_prob: float = 0.0

# ============================================================
# MODEL LOADING
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_epv_model(path):
    model = TemporalGNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.eval()
    return model

def load_pass_model(path):
    return joblib.load(path)

gnn_model = load_epv_model(GNN_EPV_PATH)
gbc_model = load_pass_model(GBC_PASS_PATH)
print('Models loaded.')

# ============================================================
# GRAPH CACHE, PASS MODEL INFERENCE, EPV CACHED INFERENCE
# ============================================================
_graph_cache = {}

def build_pass_lookup(frames_df):
    '''
    Reconstructs passes using SAME logic as training pipeline.
    Returns list of pass events with timestamps and identities.
    '''

    frames = dataframe_to_frames(frames_df)
    raw_passes = detect_passes(frames)

    lookup = []

    for p in raw_passes:
        lookup.append({
            'from': p['from'],
            'to': p['to'],
            'start_frame': p['start_frame'],
            'end_frame': p['end_frame'],
            'time': p['time'],
            'quarter': p['quarter'],
        })

    return lookup

def load_graph_cache(index_file):
    if index_file in _graph_cache:
        return _graph_cache[index_file]
    stem       = Path(index_file).stem.replace('_metadata', '')
    graph_path = Path(GRAPH_CACHE_DIR) / f'{stem}_graphs.pt'
    graphs     = torch.load(graph_path, map_location=device, weights_only=False)
    _graph_cache[index_file] = graphs
    return graphs

def save_graph_to_cache(index_file: str, sample_id: int, graph: Data):
    stem       = Path(index_file).stem.replace('_metadata', '')
    graph_path = Path(GRAPH_CACHE_DIR) / f'{stem}_graphs.pt'
    existing   = torch.load(graph_path, map_location=device, weights_only=False) \
                 if graph_path.exists() else {}
    existing[int(sample_id)] = graph
    torch.save(existing, graph_path)
    _graph_cache.pop(index_file, None)

def evaluate_pass(features):
    return gbc_model.predict_proba(features)[0, 1]

@torch.no_grad()
def evaluate_epv(index_file: str, sample_id: int) -> Optional[float]:
    graphs = load_graph_cache(index_file)
    graph  = graphs.get(int(sample_id))
    if graph is None:
        return None
    graph = graph.to(device)
    return gnn_model(graph).item()

def evaluate_sample(index_file, sample_id, pass_features=None):
    epv = evaluate_epv(index_file, sample_id)
    if pass_features is not None:
        return epv, evaluate_pass(pass_features)
    return epv

# ============================================================
# LOW-LEVEL FEATURE & FRAME BUILDING
# ============================================================

def build_frames_list(frames_df):
    ball_df   = (
        frames_df[frames_df['record_type'] == 'ball']
        .drop_duplicates('timestamp')          # ← keep first ball row per timestamp
        .set_index('timestamp')
    )
    moment_df = frames_df[frames_df['record_type'] == 'moment']

    frames_list = []

    for ts, grp in moment_df.groupby('timestamp', sort=True):
        if ts not in ball_df.index:
            continue

        br = ball_df.loc[ts]                   # now guaranteed a Series, not a DataFrame

        players = [
            {'team_id': int(r.team_id), 'player_id': int(r.player_id), 'x': r.x, 'y': r.y}
            for r in grp.itertuples() if pd.notna(r.player_id)
        ]

        if not players:
            continue

        frames_list.append({
            'timestamp': ts,
            'quarter':   int(grp['quarter'].iloc[0]),
            'time':      float(br['game_clock']),
            'ball':      {'x': float(br['x']), 'y': float(br['y']), 'z': float(br['z'])},
            'players':   players,
        })

    return frames_list

def extract_pass_features_for_gbc(
    frames_df: pd.DataFrame,
    target_ts: int,
    handler_id: int,
    receiver_id: int,
    receiver_proj_pos: np.ndarray,
    cfg: SimConfig = CFG,
) -> Optional[np.ndarray]:
    ts_arr    = np.sort(frames_df['timestamp'].unique())
    idx       = np.searchsorted(ts_arr, target_ts)
    k         = cfg.velocity_k_frames
    window_ts = ts_arr[max(0, idx - k): idx + 1]
    sub       = frames_df[frames_df['timestamp'].isin(window_ts)]

    frames_list = build_frames_list(frames_df)

    if len(frames_list) < 2:
        return None

    start_frame    = frames_list[-1]
    pass_start_pos = (start_frame['ball']['x'], start_frame['ball']['y'])
    pass_end_pos   = tuple(receiver_proj_pos[:2])

    passer_team = next(
        (p['team_id'] for p in start_frame['players'] if p['player_id'] == handler_id),
        None
    )
    if passer_team is None:
        return None

    pass_dist = distance(pass_start_pos, pass_end_pos)
    if pass_dist < 1e-3:
        return None

    pass_speed = pass_dist / (cfg.pass_delta_t_frames / cfg.fps)
    start_idx  = len(frames_list) - 1

    features = np.array([[
        pass_dist,
        pass_speed,
        compute_pass_angle(frames_list, start_idx, handler_id, pass_start_pos, pass_end_pos),
        nearest_defender_to_trajectory(start_frame, pass_start_pos, pass_end_pos, handler_id),
        max_defender_lane_depth(start_frame, pass_start_pos, pass_end_pos, handler_id),
        pass_trajectory_crowding(start_frame, pass_start_pos, pass_end_pos),
        compute_player_velocity(frames_list, start_idx, handler_id),
        passer_nearest_defender_dist(start_frame, handler_id, passer_team),
        compute_player_velocity(frames_list, start_idx, receiver_id),
        receiver_nearest_defender_dist(start_frame, pass_end_pos, passer_team),
        receiver_defender_closing_speed(start_frame, frames_list, start_idx, pass_end_pos, passer_team),
        receiver_separation_ratio(
            receiver_nearest_defender_dist(start_frame, pass_end_pos, passer_team), pass_dist
        ),
        offensive_spacing(start_frame, passer_team, handler_id),
    ]], dtype=np.float32)

    return features

# ============================================================
# CORE EPV SCORING
# ============================================================
def evaluate_epv_window(window, possession_team_id, label, model, device):
    graph = build_graph(window, possession_team_id, label)
    if graph is None:
        return None

    graph = graph.to(device)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        return model(graph).item()

@torch.no_grad()
def score_window(
    model: torch.nn.Module,
    window: pd.DataFrame,
    possession_team_id: int,
    label: float,
    device: torch.device,
) -> Optional[float]:
    return evaluate_epv_window(window, possession_team_id, label, model, device)

# ============================================================
# PASS EVALUATION CORE
# ============================================================
def evaluate_pass_option(
    window,
    frames_df,
    target_ts,
    handler_id,
    receiver_id,
    receiver_proj_pos,
    possession_team_id,
    label,
    epv_model,
    device,
    cfg
):
    synth_window = build_synthetic_window(
        window, handler_id, receiver_proj_pos, None, cfg
    )

    epv = evaluate_epv_window(
        synth_window, possession_team_id, label, epv_model, device
    )

    pass_feats = extract_pass_features_for_gbc(
        frames_df, target_ts, handler_id,
        receiver_id, receiver_proj_pos, cfg
    )

    prob = evaluate_pass(pass_feats) if pass_feats is not None else None

    return epv, prob

def simulate_pass_from_detected(
    frames_df: pd.DataFrame,
    pass_row: pd.Series,
    possession_team_id: int,
    label: float,
    epv_model: torch.nn.Module,
    device: torch.device,
    cfg: SimConfig = CFG,
) -> Optional[FrameResult]:
    window, target_ts = find_window_for_pass(
        frames_df, pass_row['time_remaining'], cfg
    )
    if window is None:
        return None

    baseline_epv = evaluate_epv_window(window, possession_team_id, label, epv_model, device)

    handler_id      = int(pass_row['from']) if 'from' in pass_row.index else None
    actual_receiver = int(pass_row['to'])   if 'to'   in pass_row.index else None

    if handler_id is None:
        handler_series = detect_handler_per_frame(frames_df, possession_team_id, cfg)
        handler_at_ts  = handler_series.get(target_ts, -1)
        if handler_at_ts == -1:
            return None
        handler_id = int(handler_at_ts)

    off_players = frames_df[
        (frames_df['record_type'] == 'moment') &
        (frames_df['team_id']     == possession_team_id) &
        (frames_df['player_id']   != handler_id)
    ]['player_id'].unique()

    if len(off_players) == 0:
        return None

    velocities    = estimate_velocities(frames_df, target_ts, cfg)
    all_projected = project_positions(frames_df, target_ts, velocities, cfg)

    receiver_epvs       = {}
    receiver_pass_probs = {}

    # --------------------------------------------------
    # Evaluate all possible pass targets
    # --------------------------------------------------
    for receiver_id in off_players:
        if receiver_id not in all_projected:
            continue

        epv, prob = evaluate_pass_option(
            window,
            frames_df,
            target_ts,
            handler_id,
            receiver_id,
            all_projected[receiver_id],
            all_projected,
            possession_team_id,
            label,
            epv_model,
            device,
            cfg
        )

        if epv is None:
            continue

        receiver_epvs[int(receiver_id)] = epv

        if prob is not None:
            receiver_pass_probs[int(receiver_id)] = float(prob)

        if not receiver_epvs:
            return None

    best_receiver_id = max(receiver_epvs, key=receiver_epvs.get)

    return FrameResult(
        timestamp            = int(target_ts),
        handler_id           = handler_id,
        baseline_epv         = baseline_epv,
        receiver_epvs        = receiver_epvs,
        receiver_pass_probs  = receiver_pass_probs,
        best_receiver_id     = best_receiver_id,
        best_delta_epv       = receiver_epvs[best_receiver_id] - baseline_epv,
        actual_next_handler  = actual_receiver,
        actual_pass_was_optimal = (
            actual_receiver == best_receiver_id
            if actual_receiver is not None else None
        ),
    )

# =============================================================
# HANDLER DETECTION, KINEMATICS, SYNTHETIC WINDOW GENERATION
# =============================================================
def detect_handler_per_frame(
    frames: pd.DataFrame,
    possession_team_id: int,
    cfg: SimConfig = CFG,
) -> pd.Series:
    ball = (
        frames[frames['record_type'] == 'ball']
        [['timestamp', 'x', 'y', 'z']]
        .rename(columns={'x': 'bx', 'y': 'by', 'z': 'bz'})
    )
    offense = frames[
        (frames['record_type'] == 'moment') &
        (frames['team_id'] == possession_team_id)
    ][['timestamp', 'player_id', 'x', 'y']]

    merged = offense.merge(ball, on='timestamp', how='inner')
    merged['dist'] = np.sqrt(
        (merged['x'] - merged['bx'])**2 +
        (merged['y'] - merged['by'])**2
    )

    closest = (
        merged.sort_values('dist')
        .groupby('timestamp', sort=False)
        .first()
        .reset_index()
        [['timestamp', 'player_id', 'bz', 'dist']]
    )
    closest.loc[closest['bz'] > cfg.z_carry_threshold, 'player_id'] = -1
    return closest.set_index('timestamp')['player_id']

def build_handler_stints(
    handler_series: pd.Series,
    cfg: SimConfig = CFG,
) -> list[dict]:
    timestamps = handler_series.index.to_numpy()
    players    = handler_series.to_numpy()

    stints           = []
    confirmed_player = -1
    confirmed_start  = None
    confirmed_end    = None
    candidate_player = -1
    candidate_count  = 0

    for ts, pid in zip(timestamps, players):
        if pid == confirmed_player:
            confirmed_end    = ts
            candidate_player = pid
            candidate_count  = 1
        else:
            if pid == candidate_player:
                candidate_count += 1
            else:
                candidate_player = pid
                candidate_count  = 1

            if candidate_count >= cfg.min_handler_frames and pid != -1:
                if (confirmed_player != -1
                        and confirmed_start is not None
                        and confirmed_end is not None):
                    n = int(confirmed_end - confirmed_start)
                    if n >= cfg.min_stint_frames:
                        stints.append({
                            'player_id': confirmed_player,
                            'start_ts':  confirmed_start,
                            'end_ts':    confirmed_end,
                            'n_frames':  n,
                        })
                confirmed_player = pid
                confirmed_start  = ts
                confirmed_end    = ts
                candidate_count  = 0

    if (confirmed_player != -1
            and confirmed_start is not None
            and confirmed_end is not None):
        n = int(confirmed_end - confirmed_start)
        if n >= cfg.min_stint_frames:
            stints.append({
                'player_id': confirmed_player,
                'start_ts':  confirmed_start,
                'end_ts':    confirmed_end,
                'n_frames':  n,
            })

    return stints

def estimate_velocities(
    frames: pd.DataFrame,
    target_ts: int,
    cfg: SimConfig = CFG,
) -> dict:
    timestamps = np.sort(frames['timestamp'].unique())
    idx        = np.searchsorted(timestamps, target_ts)
    k          = cfg.velocity_k_frames
    window_ts  = timestamps[max(0, idx - k): idx + 1]

    sub = frames[
        frames['timestamp'].isin(window_ts) &
        (frames['record_type'] == 'moment')
    ][['player_id', 'timestamp', 'x', 'y']]

    velocities = {}
    for pid, grp in sub.groupby('player_id'):
        grp = grp.sort_values('timestamp')
        if len(grp) < 2:
            velocities[pid] = np.zeros(2)
        else:
            dx = grp['x'].iloc[-1] - grp['x'].iloc[0]
            dy = grp['y'].iloc[-1] - grp['y'].iloc[0]
            dt = len(grp) - 1
            velocities[pid] = np.array([dx / dt, dy / dt])

    return velocities

def project_positions(
    frames: pd.DataFrame,
    target_ts: int,
    velocities: dict,
    cfg: SimConfig = CFG,
) -> dict:
    current = (
        frames[
            (frames['timestamp'] == target_ts) &
            (frames['record_type'] == 'moment')
        ]
        [['player_id', 'x', 'y']]
        .set_index('player_id')
    )

    return {
        pid: current.loc[pid, ['x', 'y']].to_numpy(dtype=float)
              + velocities.get(pid, np.zeros(2)) * cfg.pass_delta_t_frames
        for pid in current.index
    }

def build_ball_arc(handler_pos, receiver_pos, cfg=CFG):
    handler_pos  = np.array(handler_pos).flatten()[:2]
    receiver_pos = np.array(receiver_pos).flatten()[:2]

    T  = cfg.pass_delta_t_frames
    g  = cfg.g_ft_per_s2 / (cfg.fps ** 2)
    z0 = cfg.z_hand_height
    v0 = g * T / 2.0
    t  = np.arange(T)
    return np.stack([
        np.linspace(handler_pos[0], receiver_pos[0], T),
        np.linspace(handler_pos[1], receiver_pos[1], T),
        z0 + v0 * t - 0.5 * g * t ** 2,
    ], axis=1)

def build_synthetic_window(window, handler_id, receiver_projected, all_projected, cfg=CFG):
    T     = cfg.pass_delta_t_frames
    synth = window.copy()

    # Cast x/y/z to float64 to avoid FutureWarning on assignment
    for col in ['x', 'y', 'z']:
        if col in synth.columns:
            synth[col] = synth[col].astype(np.float64)

    receiver_projected = np.array(receiver_projected).flatten()[:2]

    timestamps = np.sort(synth['timestamp'].unique())
    if len(timestamps) < T:
        return synth

    arc_ts       = timestamps[-T:]
    arc_start_ts = arc_ts[0]

    start_rows = (
        synth[
            (synth['timestamp'] == arc_start_ts) &
            (synth['record_type'] == 'moment')
        ]
        [['player_id', 'x', 'y']]
        .set_index('player_id')
    )

    handler_start = (
        start_rows.loc[handler_id, ['x', 'y']].to_numpy(dtype=float)
        if handler_id in start_rows.index
        else receiver_projected
    )

    arc = build_ball_arc(handler_start, receiver_projected, cfg)

    for i, ts in enumerate(arc_ts):
        alpha = i / max(T - 1, 1)

        ball_mask = (synth['timestamp'] == ts) & (synth['record_type'] == 'ball')
        synth.loc[ball_mask, 'x'] = float(arc[i, 0])
        synth.loc[ball_mask, 'y'] = float(arc[i, 1])
        synth.loc[ball_mask, 'z'] = float(arc[i, 2])

        if all_projected is None:
            continue

        mom_mask = (synth['timestamp'] == ts) & (synth['record_type'] == 'moment')
        for pid, proj_pos in all_projected.items():
            proj_pos  = np.array(proj_pos).flatten()[:2].astype(float)
            p_mask = mom_mask & (synth['player_id'] == pid)
            if not p_mask.any():
                continue
            start_pos = (
                start_rows.loc[pid, ['x', 'y']].to_numpy(dtype=float).flatten()[:2]
                if pid in start_rows.index
                else proj_pos
            )
            interp_x = float(start_pos[0] + alpha * (proj_pos[0] - start_pos[0]))
            interp_y = float(start_pos[1] + alpha * (proj_pos[1] - start_pos[1]))
            synth.loc[p_mask, 'x'] = interp_x
            synth.loc[p_mask, 'y'] = interp_y

    return synth

# =============================================================
# MAIN SIMULATION LOOP
# =============================================================
def simulate_possession_passes(
    frames: pd.DataFrame,
    possession_team_id: int,
    label: float,
    epv_model: torch.nn.Module,
    pass_model,
    device: torch.device,
    cfg: SimConfig = CFG,
) -> list:
    handler_series = detect_handler_per_frame(frames, possession_team_id, cfg)
    stints         = build_handler_stints(handler_series, cfg)
    timestamps     = np.sort(frames['timestamp'].unique())
    results        = []

    for stint in stints:
        handler_id = stint['player_id']
        stint_ts   = timestamps[
            (timestamps >= stint['start_ts']) &
            (timestamps <= stint['end_ts'])
        ]

        for target_ts in stint_ts[::cfg.sample_every_n]:
            window = build_window(frames, target_ts)
            if window is None:
                continue

            baseline_epv = score_window(epv_model, window, possession_team_id, label, device)
            if baseline_epv is None:
                continue

            off_players = frames[
                (frames['record_type'] == 'moment') &
                (frames['team_id']     == possession_team_id) &
                (frames['player_id']   != handler_id)
            ]['player_id'].unique()

            if len(off_players) == 0:
                continue

            velocities    = estimate_velocities(frames, target_ts, cfg)
            all_projected = project_positions(frames, target_ts, velocities, cfg)

            receiver_epvs       = {}
            receiver_pass_probs = {}

            for receiver_id in off_players:
                if receiver_id not in all_projected:
                    continue

                synth_window = build_synthetic_window(
                    window, handler_id, all_projected[receiver_id], all_projected, cfg
                )
                epv = score_window(epv_model, synth_window, possession_team_id, label, device)
                if epv is None:
                    continue
                receiver_epvs[int(receiver_id)] = epv

                pass_feats = extract_pass_features_for_gbc(
                    frames, target_ts, handler_id,
                    receiver_id, all_projected[receiver_id], cfg
                )
                if pass_feats is not None:
                    receiver_pass_probs[int(receiver_id)] = float(evaluate_pass(pass_feats))

            if not receiver_epvs:
                continue

            best_receiver_id = max(receiver_epvs, key=receiver_epvs.get)
            results.append(FrameResult(
                timestamp            = int(target_ts),
                handler_id           = int(handler_id),
                baseline_epv         = baseline_epv,
                receiver_epvs        = receiver_epvs,
                receiver_pass_probs  = receiver_pass_probs,
                best_receiver_id     = best_receiver_id,
                best_delta_epv       = receiver_epvs[best_receiver_id] - baseline_epv,
            ))

    return results

# =============================================================
# ACTUAL PASS ANNOTATION
# =============================================================
def find_next_handler(
    handler_series: pd.Series,
    from_ts: int,
    cfg: SimConfig = CFG,
) -> Optional[int]:
    future = handler_series[
        (handler_series.index > from_ts) &
        (handler_series.index <= from_ts + cfg.handler_switch_window)
    ]
    valid = future[future != -1]
    return int(valid.iloc[0]) if not valid.empty else None

def annotate_actual_passes(
    frame_results: list,
    frames: pd.DataFrame,
    possession_team_id: int,
    cfg: SimConfig = CFG,
) -> list:
    handler_series = detect_handler_per_frame(frames, possession_team_id, cfg)
    for fr in frame_results:
        next_handler = find_next_handler(handler_series, fr.timestamp, cfg)
        if next_handler is not None and next_handler != fr.handler_id:
            fr.actual_next_handler = next_handler
    return frame_results

# =============================================================
# DETECTED PASS PIPELINE
# =============================================================
def load_detected_passes(csv_path: str, game_metadata_path: str) -> pd.DataFrame:
    df        = pd.read_csv(csv_path)
    df        = df[df['is_synthetic'] == 0]
    game_stem = Path(game_metadata_path).stem.replace('_metadata', '')
    df        = df[df['game_id'].apply(
        lambda p: Path(p).stem.replace('_metadata', '')) == game_stem
    ]
    return df.reset_index(drop=True)

def find_window_for_pass(
    frames_df: pd.DataFrame,
    game_clock_target: float,
    cfg: SimConfig = CFG,
    clock_tol: float = 0.1,
) -> tuple:
    clock_lookup = (
        frames_df[frames_df['record_type'] == 'ball']
        [['timestamp', 'game_clock']]
        .dropna(subset=['game_clock'])
        .drop_duplicates('timestamp')
        .set_index('timestamp')
    )
    if clock_lookup.empty:
        return None, None

    diffs   = (clock_lookup['game_clock'] - game_clock_target).abs()
    best_ts = diffs.idxmin()

    if diffs[best_ts] > clock_tol:
        return None, None

    window = build_window(frames_df, best_ts)
    return window, best_ts

def run_detected_pass_analysis(
    index_file: str,
    passes_csv: str,
    epv_model:  torch.nn.Module,
    device:     torch.device,
    cfg:        SimConfig = CFG,
) -> tuple:
    epv_model.eval()
    meta_df    = pd.read_parquet(index_file)
    frames_key = meta_df['frames_path'].iloc[0]
    frames_df  = pd.read_parquet(frames_key)

    pass_df = load_detected_passes(passes_csv, index_file)
    if pass_df.empty:
        print(f'No ground truth passes found for {Path(index_file).stem}')
        return pd.DataFrame(), []

    frame_results = []

    for _, pass_row in pass_df.iterrows():
        poss_match = meta_df[
            (meta_df['start_event_id'] <= pass_row.get('start_event_id', -1)) &
            (meta_df['end_event_id']   >= pass_row.get('start_event_id', -1))
        ]

        if poss_match.empty:
            possession_team_id = int(meta_df['possession_team_id'].mode().iloc[0])
            label = 0.0
        else:
            possession_team_id = int(poss_match['possession_team_id'].iloc[0])
            label              = float(poss_match['points'].iloc[0])

        start_eid   = pass_row.get('start_event_id', meta_df['start_event_id'].min())
        end_eid     = pass_row.get('end_event_id',   meta_df['end_event_id'].max())
        poss_frames = frames_df[
            frames_df['start_event_id'].between(start_eid, end_eid) |
            frames_df['end_event_id'].between(start_eid, end_eid)
        ]
        if poss_frames.empty:
            poss_frames = frames_df

        fr = simulate_pass_from_detected(
            poss_frames, pass_row, possession_team_id, label, epv_model, device, cfg
        )
        if fr is not None:
            frame_results.append(fr)

    records = [{
        'timestamp':               fr.timestamp,
        'handler_id':              fr.handler_id,
        'actual_receiver':         fr.actual_next_handler,
        'optimal_receiver':        fr.best_receiver_id,
        'is_optimal':              fr.actual_pass_was_optimal,
        'baseline_epv':            fr.baseline_epv,
        'best_delta_epv':          fr.best_delta_epv,
        'optimal_pass_prob':       fr.receiver_pass_probs.get(fr.best_receiver_id),
        'actual_pass_prob':        fr.receiver_pass_probs.get(fr.actual_next_handler),
        'all_receiver_epvs':       fr.receiver_epvs,
        'all_receiver_pass_probs': fr.receiver_pass_probs,
    } for fr in frame_results]

    return pd.DataFrame(records), frame_results

# ============================================================
# PASS EXTRACTION & ALIGNMENT
# ============================================================
def extract_detected_passes_for_possession(
    frames_df: pd.DataFrame,
    possession_team_id: int,
) -> list:
    frames_list = build_frames_list(frames_df)

    ts_arr = np.sort(frames_df['timestamp'].unique())

    raw_passes = detect_passes(frames_list)
    result     = []

    for p in raw_passes:
        dist = distance(p['start_pos'], p['end_pos'])
        dt   = abs(frames_list[p['start_frame']]['time'] - frames_list[p['end_frame']]['time'])
        spd  = dist / dt if dt > 0 else 0.0

        start_ts = (
            ts_arr[p['start_frame']]
            if p['start_frame'] < len(ts_arr)
            else ts_arr[-1]
        )

        result.append(DetectedPass(
            timestamp     = int(start_ts),
            from_player   = int(p['from']),
            to_player     = int(p['to']),
            start_pos     = p['start_pos'],
            end_pos       = p['end_pos'],
            pass_distance = float(dist),
            pass_speed    = float(spd),
            success       = check_pass_success(frames_list, p['end_frame'], p['to']),
        ))

    return result

def align_passes_with_optimal(
    detected_passes: list,
    frame_results:   list,
    ts_tolerance:    int = 15,
) -> list:
    for dp in detected_passes:
        closest_fr = min(
            frame_results,
            key=lambda fr: abs(fr.timestamp - dp.timestamp),
            default=None,
        )
        if closest_fr is None:
            continue
        if abs(closest_fr.timestamp - dp.timestamp) > ts_tolerance:
            continue
        dp.optimal_receiver      = closest_fr.best_receiver_id
        dp.aligned_with_optimal  = (dp.to_player == closest_fr.best_receiver_id)
    return detected_passes

# ============================================================
# ANALYSIS + SUMMARIZATION
# ============================================================
def summarize_possession(summary: PossessionSummary) -> PossessionSummary:
    frs = summary.frame_results
    summary.n_frames_evaluated = len(frs)

    observed = [fr for fr in frs if fr.actual_next_handler is not None]
    summary.n_passes_observed = len(observed)
    summary.n_optimal_passes  = sum(
        1 for fr in observed if fr.actual_next_handler == fr.best_receiver_id
    )

    if frs:
        summary.mean_best_delta_epv = float(np.mean([fr.best_delta_epv for fr in frs]))
    if summary.n_passes_observed > 0:
        summary.optimal_pass_rate = summary.n_optimal_passes / summary.n_passes_observed

    dps       = summary.detected_passes
    summary.n_detected_passes = len(dps)
    aligned   = [dp for dp in dps if dp.aligned_with_optimal is True]
    matchable = [dp for dp in dps if dp.aligned_with_optimal is not None]
    summary.n_aligned_passes  = len(aligned)
    summary.pass_alignment_rate = len(aligned) / len(matchable) if matchable else 0.0

    opt_probs = [
        fr.receiver_pass_probs[fr.best_receiver_id]
        for fr in frs if fr.best_receiver_id in fr.receiver_pass_probs
    ]
    summary.mean_pass_success_prob = float(np.mean(opt_probs)) if opt_probs else 0.0
    return summary

def run_analysis(
    index_file: str,
    graph_cache: dict,
    epv_model:   torch.nn.Module,
    pass_model,
    device:      torch.device,
    cfg:         SimConfig = CFG,
) -> tuple:
    epv_model.eval()
    df     = pd.read_parquet(index_file)
    graphs = graph_cache[index_file]

    frames_cache = {}
    summaries    = []

    for (frames_key, start_eid, end_eid), poss_df in df.groupby(
        ['frames_path', 'start_event_id', 'end_event_id']
    ):
        if frames_key not in frames_cache:
            raw = pd.read_parquet(frames_key)
            frames_cache[frames_key] = {
                key: grp for key, grp in raw.groupby(['start_event_id', 'end_event_id'])
            }

        poss_frames = frames_cache[frames_key].get((start_eid, end_eid))
        if poss_frames is None:
            continue

        row                = poss_df.iloc[0]
        possession_team_id = int(row['possession_team_id'])
        label              = float(row['points'])

        frame_results = simulate_possession_passes(
            poss_frames, possession_team_id, label, epv_model, pass_model, device, cfg
        )
        if not frame_results:
            continue

        frame_results = annotate_actual_passes(
            frame_results, poss_frames, possession_team_id, cfg
        )

        detected = extract_detected_passes_for_possession(poss_frames, possession_team_id)
        detected = align_passes_with_optimal(detected, frame_results)

        summary = summarize_possession(PossessionSummary(
            possession_id      = int(row.get('possession_id', -1)),
            game_path          = frames_key,
            possession_team_id = possession_team_id,
            actual_points      = label,
            frame_results      = frame_results,
            detected_passes    = detected,
        ))
        summaries.append(summary)

    frames_cache.clear()

    records = [{
        'possession_id':          s.possession_id,
        'game_path':              s.game_path,
        'possession_team_id':     s.possession_team_id,
        'actual_points':          s.actual_points,
        'n_frames_evaluated':     s.n_frames_evaluated,
        'n_passes_observed':      s.n_passes_observed,
        'n_optimal_passes':       s.n_optimal_passes,
        'optimal_pass_rate':      s.optimal_pass_rate,
        'mean_best_delta_epv':    s.mean_best_delta_epv,
        'n_detected_passes':      s.n_detected_passes,
        'n_aligned_passes':       s.n_aligned_passes,
        'pass_alignment_rate':    s.pass_alignment_rate,
        'mean_pass_success_prob': s.mean_pass_success_prob,
    } for s in summaries]

    return pd.DataFrame(records), summaries

print('Simulation cell loaded.')

# Additional Helpers
def normalize_time_remaining(df, time_col, quarter_col):
    q = df[quarter_col].astype(int)
    t = df[time_col].astype(float)

    reg_mask = q <= 4
    ot_mask  = ~reg_mask

    result = pd.Series(index=df.index, dtype=float)
    result[reg_mask] = (4 - q[reg_mask]) * QUARTER_DURATION + t[reg_mask]

    ot_num = q[ot_mask] - 4
    result[ot_mask] = -((ot_num - 1) * OVERTIME_DURATION + (OVERTIME_DURATION - t[ot_mask]))

    return result

@torch.no_grad()
def run_gnn(graph, model, device):
    if graph is None:
        return None
    graph = graph.to(device)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
    return model(graph).item()


# Driver
if __name__ == '__main__':
    # ============================================================
    # STEP 1 — Load & normalize
    # ============================================================
    print('Loading metadata and frames...')
    meta_df   = pd.read_parquet(INDEX_FILE).copy()
    frames_df = pd.read_parquet(meta_df['frames_path'].iloc[0]).copy()

    print(f'  Metadata rows : {len(meta_df)}')
    print(f'  Frames rows   : {len(frames_df)}')

    frames_df['clock_norm'] = normalize_time_remaining(
        frames_df, 'game_clock', 'quarter'
    )

    if 'game_clock' in meta_df.columns and 'quarter' in meta_df.columns:
        meta_df['clock_norm'] = normalize_time_remaining(
            meta_df, 'game_clock', 'quarter'
        )
    else:
        ball_clocks = (
            frames_df[frames_df['record_type'] == 'ball']
            [['start_event_id', 'end_event_id', 'clock_norm']]
            .groupby(['start_event_id', 'end_event_id'])
            ['clock_norm']
            .agg(clock_norm_start='max', clock_norm_end='min')
            .reset_index()
        )
        meta_df = meta_df.merge(ball_clocks, on=['start_event_id', 'end_event_id'], how='left')

    # ============================================================
    # STEP 2 — Load & normalize pass_df
    # ============================================================
    print('\nLoading detected passes...')
    pass_df = load_detected_passes(PASSES_CSV, INDEX_FILE).copy()
    pass_df = pass_df.rename(columns={
        'from_player': 'from', 'to_player': 'to',
        'passer_id':   'from', 'receiver_id': 'to',
    })

    if pass_df.empty:
        raise RuntimeError('No ground truth passes found — check game_id matching.')

    if 'game_time' in pass_df.columns:
        pass_df['clock_norm'] = pass_df['game_time'].astype(float)
    else:
        pass_df['clock_norm'] = normalize_time_remaining(
            pass_df, 'time_remaining', 'quarter'
        )

    print(f'  Ground truth passes found : {len(pass_df)}')
    print(f'  clock_norm range (passes) : [{pass_df['clock_norm'].min():.1f}, '
        f'{pass_df['clock_norm'].max():.1f}]')

    # ============================================================
    # STEP 3 — Build per-possession clock bounds from frames
    # ============================================================
    ball_frames = frames_df[frames_df['record_type'] == 'ball']

    poss_clock_bounds = (
        ball_frames
        [['start_event_id', 'end_event_id', 'clock_norm']]
        .groupby(['start_event_id', 'end_event_id'])
        ['clock_norm']
        .agg(clock_min='min', clock_max='max')
        .reset_index()
    )

    meta_df = meta_df.merge(poss_clock_bounds, on=['start_event_id', 'end_event_id'], how='left')

    print(f'\nLoading graph cache...')
    cached_graphs = load_graph_cache(INDEX_FILE)
    print(f'  Cached graphs: {len(cached_graphs)}')

    # ============================================================
    # STEP 4 — Main loop
    # ============================================================
    meta_possessions = (
        meta_df
        .drop_duplicates(subset=['start_event_id', 'end_event_id'])
        .dropna(subset=['clock_min', 'clock_max'])
        .reset_index(drop=True)
    )

    pass_clocks = pass_df['clock_norm'].values

    has_pass = meta_possessions.apply(
        lambda r: np.any(
            (pass_clocks >= r['clock_min']) & (pass_clocks <= r['clock_max'])
        ),
        axis=1
    )

    meta_filtered = meta_possessions[has_pass].reset_index(drop=True)
    print(f'Unique possessions        : {len(meta_possessions)}')
    print(f'Possessions with a pass   : {len(meta_filtered)}')


    results      = []
    skip_reasons = Counter()
    n_passes     = len(pass_df)

    for pass_idx_global, (_, pass_row) in enumerate(pass_df.iterrows()):

        if pass_idx_global % 25 == 0:
            print(f'  [{pass_idx_global:>4}/{n_passes}]  results: {len(results)}  '
                f'skipped: {sum(skip_reasons.values())}  '
                f'reasons: {dict(skip_reasons)}')

        target_clock = float(pass_row['clock_norm'])

        match = meta_filtered[
            (meta_filtered['clock_min'] <= target_clock) &
            (meta_filtered['clock_max'] >= target_clock)
        ]
        if match.empty:
            skip_reasons['no_possession_match'] += 1
            continue
        if len(match) > 1:
            match = match.loc[
                (match['clock_max'] - match['clock_min']).idxmin()
            ].to_frame().T

        meta_row  = match.iloc[0]
        start_eid = meta_row['start_event_id']
        end_eid   = meta_row['end_event_id']
        label     = float(meta_row['points'])
        is_bonus  = bool(meta_row['is_bonus'])

        frame_segment = frames_df[
            (frames_df['start_event_id'] == start_eid) &
            (frames_df['end_event_id']   == end_eid)
        ]
        if frame_segment.empty:
            skip_reasons['empty_frame_segment'] += 1
            continue

        frames_list = dataframe_to_frames(frame_segment)
        if len(frames_list) < 5:
            skip_reasons['frames_list_too_short'] += 1
            continue

        pass_events = build_pass_lookup(frame_segment)
        detected_pass_clocks = []
        for p in pass_events:
            ef = p['end_frame']
            if ef < len(frames_list):
                f = frames_list[ef]
                norm_clk = float(normalize_time_remaining(
                    pd.DataFrame([{'game_clock': f['time'], 'quarter': f['quarter']}]),
                    'game_clock', 'quarter'
                ).iloc[0])
                detected_pass_clocks.append((norm_clk, p))

        if not detected_pass_clocks:
            skip_reasons['no_detected_passes'] += 1
            continue

        best_clk, actual_pass = min(
            detected_pass_clocks, key=lambda x: abs(x[0] - target_clock)
        )
        if abs(best_clk - target_clock) > 2.0:
            skip_reasons['clock_mismatch'] += 1
            continue

        handler_id      = actual_pass['from']
        actual_receiver = actual_pass['to']
        pass_idx        = actual_pass['end_frame']

        # KEY FIX: derive offense from handler's team_id, not possession_team_id
        handler_frame = frames_list[pass_idx]
        handler_team_id = next(
            (p['team_id'] for p in handler_frame['players'] if p['player_id'] == handler_id),
            None
        )
        if handler_team_id is None:
            skip_reasons['handler_team_not_found'] += 1
            continue

        # Use handler_team_id for both GNN graph construction and candidate filtering
        possession_team_id = handler_team_id

        segment_timestamps = np.sort(frame_segment['timestamp'].unique())
        pass_idx_clamped   = min(pass_idx, len(segment_timestamps) - 1)
        window_start_idx   = max(0, pass_idx_clamped - (WINDOW_SIZE - 1))
        window_ts_slice    = segment_timestamps[window_start_idx : pass_idx_clamped + 1]

        if len(window_ts_slice) < WINDOW_SIZE:
            skip_reasons['window_too_short'] += 1
            continue

        window_df = frame_segment[frame_segment['timestamp'].isin(window_ts_slice)]
        if window_df.empty:
            skip_reasons['empty_window_df'] += 1
            continue

        target_ts_for_velocity = window_ts_slice[-1]
        velocities    = estimate_velocities(window_df, target_ts_for_velocity, CFG)
        all_projected = project_positions(window_df, target_ts_for_velocity, velocities, CFG)

        # Candidates: teammates only, max 4 (5 on court minus handler) 
        off_players = [
            p['player_id']
            for p in handler_frame['players']
            if p['team_id'] == handler_team_id
            and p['player_id'] != handler_id
        ]

        if not off_players:
            skip_reasons['no_off_players'] += 1
            continue

        # Actual receiver must be a teammate — skip if not 
        actual_receiver_team = next(
            (p['team_id'] for p in handler_frame['players'] if p['player_id'] == actual_receiver),
            None
        )
        if actual_receiver_team != handler_team_id:
            skip_reasons['actual_receiver_not_teammate'] += 1
            continue

        receiver_results = {}

        for receiver_id in off_players:
            if receiver_id not in all_projected:
                continue

            proj_pos     = all_projected[receiver_id]
            synth_window = build_synthetic_window(window_df, handler_id, proj_pos, all_projected, CFG)
            synth_graph  = build_graph(synth_window, possession_team_id, label, is_bonus, get_stat_map())
            if synth_graph is None:
                continue

            future_epv = run_gnn(synth_graph, gnn_model, device)
            pass_feats = extract_pass_features_for_gbc(
                window_df, target_ts_for_velocity, handler_id, receiver_id, proj_pos, CFG
            )
            pass_prob    = float(evaluate_pass(pass_feats)) if pass_feats is not None else None
            expected_epv = pass_prob * future_epv            if pass_prob is not None else None

            receiver_results[int(receiver_id)] = {
                'future_epv':   future_epv,
                'pass_prob':    pass_prob,
                'expected_epv': expected_epv,
            }

        valid_receivers = {
            r: v for r, v in receiver_results.items()
            if v['expected_epv'] is not None
        }
        if not valid_receivers:
            skip_reasons['no_valid_receivers'] += 1
            continue

        if actual_receiver not in receiver_results:
            skip_reasons['actual_receiver_not_in_candidates'] += 1
            continue

        best_receiver_id = max(valid_receivers, key=lambda r: valid_receivers[r]['expected_epv'])

        baseline_graph = build_graph(window_df, possession_team_id, label, is_bonus, get_stat_map())
        baseline_epv   = run_gnn(baseline_graph, gnn_model, device)

        results.append({
            'start_event_id':       start_eid,
            'end_event_id':         end_eid,
            'clock_norm':           target_clock,
            'handler_id':           handler_id,
            'actual_receiver':      actual_receiver,
            'optimal_receiver':     best_receiver_id,
            'is_optimal':           actual_receiver == best_receiver_id,
            'baseline_epv':         baseline_epv,
            'actual_expected_epv':  receiver_results.get(actual_receiver, {}).get('expected_epv'),
            'optimal_expected_epv': valid_receivers[best_receiver_id]['expected_epv'],
            'receiver_results':     receiver_results,
        })

    # After the main loop, before Step 5
    results_df = pd.DataFrame(results)
    results_df = results_df.drop_duplicates(
        subset=['start_event_id', 'end_event_id', 'clock_norm', 'handler_id']
    )
    results = results_df.to_dict('records')
    print(f'After dedup: {len(results)} unique passes')

    # ============================================================
    # STEP 5 — Summarize
    # ============================================================
    results_df = pd.DataFrame(results)

    print(f'\n{'='*50}')
    print(f'Passes evaluated : {len(results)}')
    print(f'\nSkip reason breakdown:')
    for reason, count in skip_reasons.most_common():
        print(f'  {reason:<35} : {count}')

    if len(results_df) > 0:
        n_matchable = results_df['actual_receiver'].notna().sum()
        n_optimal   = results_df['is_optimal'].sum()
        if n_matchable > 0:
            print(f'Optimal pass rate  : {n_optimal}/{n_matchable} '
                f'({100 * n_optimal / n_matchable:.1f}%)')
        print(f'Mean baseline EPV  : {results_df['baseline_epv'].mean():.4f}')
        print(f'Mean optimal EPV   : {results_df['optimal_expected_epv'].mean():.4f}')
        print(f'Mean actual EPV    : {results_df['actual_expected_epv'].dropna().mean():.4f}')
        print('\nSample output:')
        print(results_df[[
            'start_event_id', 'end_event_id', 'clock_norm',
            'handler_id', 'actual_receiver', 'optimal_receiver',
            'is_optimal', 'baseline_epv',
            'actual_expected_epv', 'optimal_expected_epv',
        ]].head(10).to_string(index=False))

    # ============================================================
    # STEP 6 — Save results with per-receiver EPV detail
    # ============================================================
    receiver_rows = []

    for r in results:
        base = {
            'start_event_id':       r['start_event_id'],
            'end_event_id':         r['end_event_id'],
            'clock_norm':           r['clock_norm'],
            'handler_id':           r['handler_id'],
            'actual_receiver':      r['actual_receiver'],
            'optimal_receiver':     r['optimal_receiver'],
            'is_optimal':           r['is_optimal'],
            'baseline_epv':         r['baseline_epv'],
            'actual_expected_epv':  r['actual_expected_epv'],
            'optimal_expected_epv': r['optimal_expected_epv'],
        }

        for receiver_id, vals in r['receiver_results'].items():
            if vals['expected_epv'] is None:
                continue
            receiver_rows.append({
                **base,
                'candidate_receiver_id':  receiver_id,
                'candidate_future_epv':   vals['future_epv'],
                'candidate_pass_prob':    vals['pass_prob'],
                'candidate_expected_epv': vals['expected_epv'],
                'is_actual_receiver':     receiver_id == r['actual_receiver'],
                'is_optimal_receiver':    receiver_id == r['optimal_receiver'],
            })

    detail_df = pd.DataFrame(receiver_rows)
    detail_df.to_csv(SIMULATION_RESULTS_PATH, index=False)

    print(f'\nSaved {len(detail_df)} rows ({len(results)} passes × avg '
        f'{len(detail_df)/max(len(results),1):.1f} receivers) → {SIMULATION_RESULTS_PATH}')
    print(detail_df.head(10).to_string(index=False))

