import os
import json
import pandas as pd
from pathlib import Path
import gc

import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

from utils.pass_features import build_pass_dataset, generate_synthetic_passes, get_closest_player, distance

# File Paths
BASE_DIR = Path.cwd()
PARQUET_DIR = BASE_DIR / 'NBA_SportVU_Parquet'
CACHE_DIR = BASE_DIR / 'Dataset_Cache'
GRAPH_CACHE_DIR = BASE_DIR / 'Graph_Cache'
MODEL_DATA_DIR = BASE_DIR / 'Model_Data'

JSON_PATH = MODEL_DATA_DIR / 'data_splits.json'
OUTPUT_FILE = BASE_DIR / 'all_games_passes_3.0.csv'
MODEL_PATH = MODEL_DATA_DIR / 'gbc_pass_model_real_only.joblib'


# Global Variables
POSSESSION_RADIUS = 5.0
MIN_PASS_DISTANCE = 2.0
MIN_BALL_SPEED = 3.0
MIN_POSSESSION_FRAMES = 3
SUCCESS_POSSESSION_FRAMES = 5


# =======================
# LOAD CACHED DATA
# =======================
def load_game_data(game_name):
    stem = Path(game_name).stem
    for suffix in ('_metadata', '_frames'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break

    metadata_path = os.path.join(CACHE_DIR, f'{stem}_metadata.parquet')
    frames_path   = os.path.join(CACHE_DIR, f'{stem}_frames.parquet')

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'No metadata cache for {stem}')
    if not os.path.exists(frames_path):
        raise FileNotFoundError(f'No frames cache for {stem}')

    meta_df    = pd.read_parquet(metadata_path)
    frames_all = pd.read_parquet(frames_path).sort_values(['start_event_id', 'end_event_id', 'timestamp'])

    return meta_df, frames_all

def dataframe_to_frames(df):
    ball_df = (
        df[df['record_type'] == 'ball']
        .drop_duplicates('timestamp')   
        .set_index('timestamp')
    )
    moment_df = df[df['record_type'] == 'moment']

    frames = []

    for ts, player_group in moment_df.groupby('timestamp', sort=True):
        if ts not in ball_df.index:
            continue

        ball_row = ball_df.loc[ts]             # now always a Series

        players = [
            {'team_id': int(r.team_id), 'player_id': int(r.player_id), 'x': r.x, 'y': r.y}
            for r in player_group.itertuples()
            if pd.notna(r.player_id)
        ]

        if not players:
            continue

        frames.append({
            'quarter': int(player_group['quarter'].iloc[0]),
            'time':    float(ball_row['game_clock']),
            'ball':    {'x': float(ball_row['x']), 'y': float(ball_row['y']), 'z': float(ball_row['z'])},
            'players': players,
        })

    return frames

# =======================
# PASS DETECTION
# =======================
def detect_passes(frames):

    passes = []

    prev_ball_pos = None
    prev_time = None

    current_handler = None
    current_team = None
    possession_count = 0

    last_stable_handler = None
    last_stable_team = None

    ball_free = False
    pass_start_pos = None
    pass_start_time = None
    pass_start_idx = None

    for i, frame in enumerate(frames):

        ball_pos = (frame['ball']['x'], frame['ball']['y'])
        players = frame['players']
        time = frame['time']
        quarter = frame['quarter']

        closest_id, closest_dist = get_closest_player(ball_pos, players)

        if closest_dist < POSSESSION_RADIUS:
            new_handler = closest_id
            new_team = next(
                p['team_id'] for p in players if p['player_id'] == closest_id
            )
        else:
            new_handler = None
            new_team = None

        if new_handler == current_handler and new_handler is not None:
            possession_count += 1
        else:
            possession_count = 1
            current_handler = new_handler
            current_team = new_team

        if possession_count >= MIN_POSSESSION_FRAMES and current_handler is not None:
            stable_handler = current_handler
            stable_team = current_team
        else:
            stable_handler = None
            stable_team = None

        if prev_ball_pos is not None and prev_time is not None:
            dt = abs(time - prev_time)
            ball_speed = distance(ball_pos, prev_ball_pos) / dt if dt > 0 else 0
        else:
            ball_speed = 0

        # PASS START
        if last_stable_handler is not None and stable_handler is None:
            ball_free = True
            pass_start_pos = prev_ball_pos
            pass_start_time = prev_time
            pass_start_idx = i - 1

        # PASS END
        if ball_free and stable_handler is not None:
            if last_stable_handler is not None:
                if stable_handler != last_stable_handler and stable_team == last_stable_team:

                    move_dist = distance(pass_start_pos, ball_pos) if pass_start_pos else 0
                    dt = abs(time - pass_start_time) if pass_start_time else 0
                    speed = move_dist / dt if dt > 0 else 0

                    if move_dist > MIN_PASS_DISTANCE and speed > MIN_BALL_SPEED:
                        passes.append({
                            'from': last_stable_handler,
                            'to': stable_handler,
                            'start_frame': pass_start_idx,
                            'end_frame': i,
                            'quarter': quarter,
                            'time': time,
                            'start_pos': pass_start_pos,
                            'end_pos': ball_pos
                        })

            ball_free = False

        if stable_handler is not None:
            last_stable_handler = stable_handler
            last_stable_team = stable_team

        prev_ball_pos = ball_pos
        prev_time = time

    return passes

# =======================
# EXECUTE ALGORITHM, BUILD PASS FEATURES
# =======================
def process_all_games_parquet(game_list):

    global_pass_id = 0
    first_write = True
    failed_games = []

    for game_name in game_list:

        print(f'Processing {game_name}...')

        try:
            meta_df, frames_all = load_game_data(game_name)
            all_passes = []

            for (start_eid, end_eid), df_segment in frames_all.groupby(['start_event_id', 'end_event_id'], sort=False):
                frames = dataframe_to_frames(df_segment)

                if len(frames) < 5:
                    continue

                passes = detect_passes(frames)

                dataset = build_pass_dataset(frames, passes, game_name, global_pass_id)
                global_pass_id += len(dataset)
                all_passes.extend(dataset)

                synth_dataset, global_pass_id = generate_synthetic_passes(
                    frames, passes, game_name, global_pass_id, k_negatives=1
                )
                all_passes.extend(synth_dataset)

            del meta_df, frames_all
            gc.collect()

            if not all_passes:
                continue

            df = pd.DataFrame(all_passes)
            df.to_csv(OUTPUT_FILE, mode='a', header=first_write, index=False)
            first_write = False

        except Exception as e:
            print(f'Failed on {game_name}: {e}')
            failed_games.append(game_name)

    print('Done.')
    print('Failed games:', failed_games)


if __name__ == '__main__':
    # Make all directories
    for d in [PARQUET_DIR, CACHE_DIR, GRAPH_CACHE_DIR, MODEL_DATA_DIR]:
        os.makedirs(d, exist_ok=True)

    # Importing File Splits
    with open(JSON_PATH, 'r') as f:
        json_data = json.load(f)

    train_files = json_data['train']
    val_files = json_data['val']
    test_files = json_data['test']
    all_files = train_files + val_files + test_files

    # Processes and outputs to OUTPUT_FILE
    process_all_games_parquet(all_files)
    df = pd.read_csv(OUTPUT_FILE)

    # Real passes only, game-level splits
    real_df = df[df['is_synthetic'] == 0]
    train_df = real_df[real_df['game_id'].isin(train_files)]
    test_df  = real_df[real_df['game_id'].isin(test_files)]

    feature_cols = [
        'pass_distance',
        'pass_angle',
        'nearest_defender_dist',
        'max_defender_lane_depth',
        'pass_trajectory_crowding',
        'passer_velocity',
        'passer_nearest_defender_dist',
        'receiver_velocity',
        'receiver_nearest_defender_dist',
        'receiver_defender_closing_speed',
        'receiver_separation_ratio',
        'offensive_spacing',
        'defenders_in_lane'
    ]

    X_train, y_train = train_df[feature_cols].values, train_df['success'].values
    X_test,  y_test  = test_df[feature_cols].values,  test_df['success'].values

    # Model + grid 
    gbc = GradientBoostingClassifier(random_state=7)
    param_grid = {
        'n_estimators':  [300, 400, 500],
        'learning_rate': [0.05, 0.1],
        'max_depth':     [2, 3]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    scoring = {
        'accuracy': 'accuracy',
        'log_loss': 'neg_log_loss'
    }

    # Balanced sample weights
    w_train = compute_sample_weight('balanced', y_train)
    grid_clean = GridSearchCV(
        estimator=gbc,
        param_grid=param_grid,
        scoring=scoring,
        refit='log_loss',
        cv=cv,
        verbose=2,
        n_jobs=-1
    )
    grid_clean.fit(X_train, y_train, sample_weight=w_train)
    print('Best params:', grid_clean.best_params_)
    print('Best (neg) log loss:', grid_clean.best_score_)

    # Save 
    joblib.dump(grid_clean.best_estimator_, MODEL_PATH)
    print(f'Model saved to: {MODEL_PATH}')

