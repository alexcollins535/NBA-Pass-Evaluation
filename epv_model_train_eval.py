from pathlib import Path
import os
import json
import pandas as pd
import numpy as np
import time
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Batch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset


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

# Model Training Configuration
NUM_EPOCHS_FULL = 10
NUM_EPOCHS_LATE = 3

LATE_SECONDS = 5.0
LR_FINETUNE = 1e-5          

# Create Dataset Class
class PossessionDataset(Dataset):
    def __init__(self, index_file: str, graphs: dict):
        self.index = pd.read_parquet(index_file).reset_index(drop=True)
        self.graphs = graphs
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sample_id = int(self.index.iloc[idx]['sample_id'])
        return self.graphs.get(sample_id, None)

    def clear_frames_cache(self):
        pass

# Helpers
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return Batch.from_data_list(batch)

def get_graph_cache(index_file):
    '''Load a single game's graph cache from disk, on demand.'''
    stem = Path(index_file).stem.replace('_metadata', '')
    graph_path = Path(GRAPH_CACHE_DIR) / f'{stem}_graphs.pt'
    return torch.load(graph_path, weights_only=False)

def count_possession_remaining(index_files, seconds_remaining):
    '''Count samples within `seconds_remaining` seconds of possession end.'''
    total = 0
    poss_key = ['start_event_id', 'end_event_id', 'possession_team_id', 'quarter']
    for f in index_files:
        meta = pd.read_parquet(f)
        meta['poss_end_ts'] = meta.groupby(poss_key)['timestamp'].transform('max')
        meta['sec_to_end']  = (meta['poss_end_ts'] - meta['timestamp']) / 1000.0
        total += (meta['sec_to_end'] <= seconds_remaining).sum()
    return total

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path=CHECKPOINT_PATH):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, path)
    print(f'  Checkpoint saved at epoch {epoch}')

def load_checkpoint(model, optimizer, path=CHECKPOINT_PATH):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'  Resumed from epoch {checkpoint['epoch']} '
            f'(val loss: {checkpoint['val_loss']:.4f})')
        return start_epoch
    return 1  # No checkpoint found, start fresh

# Main Training Loop
def train_on_games(model, optimizer, criterion, game_files, batch_size=256):
    model.train()
    total_loss, total_graphs = 0, 0
    for game_num, index_file in enumerate(game_files, 1):
        t0 = time.time()

        # Load this game's cache on demand, discard after
        game_graphs = get_graph_cache(index_file)
        ds = PossessionDataset(index_file, game_graphs)

        print(f'  [{game_num}/{len(game_files)}] {index_file}')
        print(f'    samples in file: {len(ds)}')

        loader = TorchDataLoader(ds, batch_size=batch_size, shuffle=True,
                                 num_workers=0, collate_fn=collate_skip_none)
        batch_count = 0
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs
            batch_count += 1
            if batch_count % 100 == 0:
                elapsed = time.time() - t0
                print(f'    batch {batch_count}, elapsed: {elapsed:.1f}s')

        ds.clear_frames_cache()
        del loader, ds, game_graphs  # explicitly release graph cache
        gc.collect()

        elapsed = time.time() - t0
        print(f'    done — {total_graphs} samples so far, file took {elapsed:.1f}s')
    return total_loss / total_graphs

# Main Evaluation Loop
@torch.no_grad()
def eval_on_games(model, game_files, batch_size=32):
    model.eval()
    total_loss, total_graphs = 0, 0
    for index_file in game_files:
        game_graphs = get_graph_cache(index_file)
        ds = PossessionDataset(index_file, game_graphs)
        loader = TorchDataLoader(ds, batch_size=batch_size, shuffle=False,
                                 num_workers=0, collate_fn=collate_skip_none)
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(device)
            total_loss += criterion(model(batch), batch.y.view(-1)).item() * batch.num_graphs
            total_graphs += batch.num_graphs
        ds.clear_frames_cache()
        del loader, ds, game_graphs
        gc.collect()
    return total_loss / total_graphs

# Late Possession Training Loop
def train_on_games_possession_remaining(model, optimizer, criterion, game_files, seconds_remaining=5.0, batch_size=256):
    '''
    Fine-tune on samples within `seconds_remaining` seconds of possession end
    '''
    
    model.train()
    total_loss = 0
    total_graphs = 0
    poss_key = ['start_event_id', 'end_event_id', 'possession_team_id', 'quarter']

    for game_num, index_file in enumerate(game_files, 1):
        t0 = time.time()

        # Filter to late-possession sample IDs
        meta = pd.read_parquet(index_file)
        meta['poss_end_ts'] = meta.groupby(poss_key)['timestamp'].transform('max')
        meta['sec_to_end']  = (meta['poss_end_ts'] - meta['timestamp']) / 1000.0
        late_ids = set(meta.loc[meta['sec_to_end'] <= seconds_remaining,
                                'sample_id'].astype(int))

        if not late_ids:
            print(f'  [{game_num}/{len(game_files)}] {index_file} — no late samples, skipping')
            continue

        game_graphs = get_graph_cache(index_file)
        ds = PossessionDataset(index_file, game_graphs)

        # Restrict the dataset index to late-possession rows only
        ds.index = (ds.index[ds.index['sample_id'].astype(int).isin(late_ids)]
                      .reset_index(drop=True))

        print(f'  [{game_num}/{len(game_files)}] {index_file}')
        print(f'    late samples: {len(ds)} / {len(meta)} total')

        loader = TorchDataLoader(ds, batch_size=batch_size, shuffle=True,
                                 num_workers=0, collate_fn=collate_skip_none)
        batch_count = 0
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss  += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs
            batch_count  += 1
            if batch_count % 100 == 0:
                print(f'    batch {batch_count}, elapsed: {time.time()-t0:.1f}s')

        ds.clear_frames_cache()
        del loader, ds, game_graphs
        gc.collect()

        print(f'    done — {total_graphs} late samples so far, '
              f'file took {time.time()-t0:.1f}s')

    return total_loss / total_graphs if total_graphs > 0 else float('nan')

# Late Possession Evaluation Loop
@torch.no_grad()
def eval_on_games_possession_remaining(model, game_files, seconds_remaining=3.0, batch_size=32):
    '''Evaluate only on samples within `seconds_remaining` seconds of possession end.'''
    model.eval()
    total_loss, total_graphs = 0, 0

    for index_file in game_files:
        meta = pd.read_parquet(index_file)

        poss_key = ['start_event_id', 'end_event_id', 'possession_team_id', 'quarter']
        meta['poss_end_ts'] = meta.groupby(poss_key)['timestamp'].transform('max')
        meta['ms_to_end']   = meta['poss_end_ts'] - meta['timestamp']
        meta['sec_to_end']  = meta['ms_to_end'] / 1000.0

        late_ids = set(meta.loc[meta['sec_to_end'] <= seconds_remaining, 'sample_id'].astype(int))

        if not late_ids:
            continue

        game_graphs = get_graph_cache(index_file)
        ds = PossessionDataset(index_file, game_graphs)
        ds.index = ds.index[ds.index['sample_id'].astype(int).isin(late_ids)].reset_index(drop=True)

        loader = TorchDataLoader(ds, batch_size=batch_size, shuffle=False,
                                 num_workers=0, collate_fn=collate_skip_none)
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(device)
            total_loss += criterion(model(batch), batch.y.view(-1)).item() * batch.num_graphs
            total_graphs += batch.num_graphs

        ds.clear_frames_cache()
        del loader, ds, game_graphs
        gc.collect()

    return total_loss / total_graphs if total_graphs > 0 else float('nan')

# Model
class TemporalGNN(nn.Module):
    def __init__(self, dyn_dim=3, static_dim=7, hidden_dim=128, heads=4):
        super().__init__()

        # LSTM takes a sequence and processes it step by step, maintaining a hidden state
        # that accumulates context as it moves through the sequence. At each of the 25
        # timesteps, it reads the 6 channel values for a node and updates its internal state
        # We use the output h_n, a summary of the complete trajectory in sequence
        self.lstm = nn.LSTM(dyn_dim, hidden_dim, batch_first=True)

        # Static encoder (player skill + height)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        self.static_gate = nn.Linear(32, hidden_dim)

        # Fusion layer
        self.fuse = nn.Linear(hidden_dim, hidden_dim)


        # GATv2Conv performs graph convolution with attention. For each node it looks at
        # all its neighbors, computes an attention score for each neighbor based on both
        # feature vectors and the edge attributes, then produces a weighted sum of neighbor
        # features. The intuition here is that a defender 2 feet away gets a higher attention
        # score than one 20 feet away.
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=1)

        # Standard MLP taking the pooled tensor to the final scalar output.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):

        # Temporal encoding
        x_dyn = data.x.view(-1, WINDOW_SIZE, 3)
        _, (h_n, _) = self.lstm(x_dyn)
        h_dyn = h_n.squeeze(0)

        # Static encoding
        h_static = self.static_mlp(data.x_static)

        # Fusion
        gate = torch.sigmoid(self.static_gate(h_static))
        # For each feature, how much should I keep or suppress the dynamic
        # movement signal given these are their stats?
        h = h_dyn * gate
        h = self.fuse(h)

        # GNN
        h = F.relu(self.conv1(h, data.edge_index, edge_attr=data.edge_attr))
        h = F.relu(self.conv2(h, data.edge_index, edge_attr=data.edge_attr))

        # Pooling (exclude basket)
        mask = torch.ones(h.shape[0], dtype=torch.bool, device=h.device)
        mask[11::N_NODES] = False

        h_pool = h[mask]
        batch_pool = data.batch[mask]

        h_pool = global_mean_pool(h_pool, batch_pool)
        h_pool = torch.cat([h_pool, data.u], dim=-1)

        return self.mlp(h_pool).squeeze(-1)

# Driver to train and evaluate GNN Model
def setup_train_eval_model():
    '''
    Complete model setup
    Run model training and evaluation
    '''

    # Load train/test split
    with open(SPLIT_SAVE_PATH, 'r') as f:
        json_data = json.load(f)

    train_files = json_data['train']
    val_files = json_data['val']
    test_files = json_data['test']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Integrity Check
    test_cache = get_graph_cache(train_files[0])
    sample_ds = PossessionDataset(train_files[0], test_cache)
    model = TemporalGNN(dyn_dim=3, static_dim=7, hidden_dim=128).to(device)
    del sample_ds, test_cache
    gc.collect()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Main Training Loop
    start_epoch = load_checkpoint(model, optimizer)

    for epoch in range(start_epoch, NUM_EPOCHS_FULL + 1):
        train_loss = train_on_games(model, optimizer, criterion, train_files)
        val_loss = eval_on_games(model, val_files)
        print(f'Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}')
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss)


    # Fine Tuning Training Loop
    # Swap in a lower learning rate without resetting momentum/state
    for pg in optimizer.param_groups:
        pg['lr'] = LR_FINETUNE

    start_epoch_late = load_checkpoint(model, optimizer, path=CHECKPOINT_LATE_PATH)

    for epoch in range(start_epoch_late, NUM_EPOCHS_LATE + 1):
        train_loss = train_on_games_possession_remaining(model, optimizer, criterion, train_files, seconds_remaining=LATE_SECONDS)
        val_loss = eval_on_games_possession_remaining(model, val_files, seconds_remaining=LATE_SECONDS)
        print(f'Late epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}')
        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, path=CHECKPOINT_LATE_PATH)


    # Evaluation on Test Set
    test_loss = eval_on_games(model, test_files)
    print(f'\nTest Loss (full): {test_loss:.4f}')

    for seconds in [10.0, 8.0, 2.0, 4.0, 5.0, 6.0]:
        loss = eval_on_games_possession_remaining(model, test_files, seconds_remaining=seconds)
        n = count_possession_remaining(test_files, seconds)
        print(f'  ≤{seconds:.1f}s to end: loss={loss:.4f}  (n={n})')

    torch.save(model.state_dict(), MODEL_LATE_PATH)
    print(f'Fine-tuned model saved to {MODEL_LATE_PATH}')


if __name__ == '__main__':
    setup_train_eval_model()