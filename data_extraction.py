import requests
import json
import tempfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.notebook import tqdm
from nba_api.stats.endpoints import commonallplayers
import pandas as pd
import unicodedata
import re
import os
from pathlib import Path

# File Paths
BASE_DIR = Path.cwd()
OUTPUT_DIR = BASE_DIR / 'NBA_SportVU_Parquet'

SHOOTING_DATA_FILEPATH = BASE_DIR / 'shooting_data.csv'
FILTERED_SHOOTING_FILEPATH = BASE_DIR / 'shooting_data_filtered.csv'

# Import URLs
API_URL = (
    'https://api.github.com/repos/linouk23/NBA-Player-Movements' '/contents/data/2016.NBA.Raw.SportVU.Game.Logs'
)
PBP_URL = 'https://github.com/sumitrodatta/nba-alt-awards/raw/main/Historical/PBP%20Data/2015-16_pbp.csv'


# =============================================
# Part 1: Convert Files from Raw to Parquet
# =============================================
def get_file_list_from_github():
    '''
    Scrape file list from GitHub API
    '''
    resp = requests.get(API_URL, headers={'Accept': 'application/vnd.github.v3+json'})
    resp.raise_for_status()
    items = resp.json()

    files = [
        {
            'name': item['name'],
            'download_url': item['download_url'],
            'stem': item['name'].replace('.7z', '')
        }
        for item in items
        if item['name'].endswith('.7z')
    ]

    print(f'Found {len(files)} game files')
    return files

def get_event_data_from_github():
    '''
    Download play-by-play for event details
    Provides event type, possession team, home/away descriptions.
    Source: same PBP CSV used in the HuggingFace dataset script.
    '''
    try:
        pbp = pd.read_csv(PBP_URL)
        pbp['GAME_ID'] = pbp['GAME_ID'].astype(int)
        pbp['EVENTNUM'] = pbp['EVENTNUM'].astype(int)
        print(f'PBP loaded: {pbp.shape[0]:,} rows')
        has_pbp = True
    except Exception as e:
        print(f'PBP load failed (events will have no enrichment): {e}')
        pbp = None
        has_pbp = False
    return pbp, has_pbp

def download_and_extract_json(download_url, name):
    '''
    Download a .7z archive, extract it, return parsed JSON.
    '''
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, name)
        r = requests.get(download_url, stream=True, timeout=120)
        r.raise_for_status()
        with open(archive_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                f.write(chunk)
        sp.run(['7z', 'e', archive_path, f'-o{tmpdir}', '-y', '-bd'],
               capture_output=True)
        json_files = [fp for fp in os.listdir(tmpdir) if fp.endswith('.json')]
        if not json_files:
            raise ValueError(f'No JSON found in {name}')
        with open(os.path.join(tmpdir, json_files[0]), 'r') as f:
            data = json.load(f)
    return data

def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def parse_game(data, stem, pbp=None):
    '''
    Parse one game's SportVU JSON into three DataFrames.

    moments_df  — player tracking (one row per player per moment)
    ball_df     — ball tracking   (one row per moment)
    events_df   — event metadata, optionally enriched from PBP
    '''
    game_id = data.get('gameid', stem)
    game_date = data.get('gamedate', '')

    moment_rows = []
    ball_rows   = []
    event_rows  = []

    for event in data.get('events', []):
        event_id = event.get('eventId')

        # Event metadata
        event_meta = {
            'game_id': game_id,
            'game_date': game_date,
            'event_id': event_id,
            'home_team_id': event.get('home', {}).get('teamid'),
            'visitor_team_id': event.get('visitor', {}).get('teamid'),
            'home_abbrev': event.get('home', {}).get('abbreviation'),
            'visitor_abbrev': event.get('visitor', {}).get('abbreviation'),
            # PBP-enriched fields (populated below if available)
            'event_type': None,
            'possession_team_id': None,
            'desc_home': None,
            'desc_away': None,
        }

        if pbp is not None:
            try:
                row = pbp.loc[
                    (pbp.GAME_ID == int(game_id)) &
                    (pbp.EVENTNUM == int(event_id))
                ]
                if len(row) == 1:
                    event_meta['event_type'] = row['EVENTMSGTYPE'].item()
                    event_meta['desc_home']  = str(row['HOMEDESCRIPTION'].item())
                    event_meta['desc_away']  = str(row['VISITORDESCRIPTION'].item())
                    # Possession: offensive events (1-5) -> PLAYER1 team; foul (6) -> PLAYER2
                    etype = int(row['EVENTMSGTYPE'].item())
                    if etype in (1, 2, 3, 4, 5):
                        event_meta['possession_team_id'] = row['PLAYER1_TEAM_ID'].item()
                    elif etype == 6:
                        event_meta['possession_team_id'] = row['PLAYER2_TEAM_ID'].item()
            except Exception:
                pass

        # Moments
        for moment in event.get('moments', []):
            if len(moment) < 6 or not moment[5]:
                continue

            quarter = moment[0]
            timestamp = moment[1]
            game_clock = safe_float(moment[2])
            shot_clock = safe_float(moment[3])
            entities = moment[5]

            # Ball is always the first entity (team_id = -1, player_id = -1)
            ball = entities[0]
            ball_rows.append({
                'game_id': game_id,
                'event_id': event_id,
                'quarter':  quarter,
                'timestamp': timestamp,
                'game_clock': game_clock,
                'shot_clock': shot_clock,
                'x': safe_float(ball[2]),
                'y': safe_float(ball[3]),
                'z': safe_float(ball[4]),
            })

            # Players are all entities after the ball
            for p in entities[1:]:
                moment_rows.append({
                    'game_id': game_id,
                    'event_id': event_id,
                    'quarter': quarter,
                    'timestamp': timestamp,
                    'game_clock': game_clock,
                    'shot_clock': shot_clock,
                    'team_id': p[0],
                    'player_id': p[1],
                    'x': safe_float(p[2]),
                    'y': safe_float(p[3]),
                    'z': safe_float(p[4]),
                })

        first_moment = next(
            (m for m in event.get('moments', []) if len(m) >= 6 and m[5]),
            None
        )
        if first_moment:
            event_meta['quarter']    = first_moment[0]
            event_meta['game_clock'] = safe_float(first_moment[2])
            event_meta['shot_clock'] = safe_float(first_moment[3])
            event_meta['timestamp']  = first_moment[1]

        event_rows.append(event_meta)

    moments_df = pd.DataFrame(moment_rows)
    ball_df = pd.DataFrame(ball_rows)
    events_df = pd.DataFrame(event_rows)

    # Downcast numeric columns to save space
    for df in (moments_df, ball_df, events_df):
        if df.empty:
            continue
        for col in ('game_clock', 'shot_clock', 'x', 'y', 'z'):
            if col in df.columns:
                df[col] = df[col].astype('float32')
        if 'quarter' in df.columns:
            df['quarter'] = pd.array(df['quarter'], dtype='Int8')
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.array(df['timestamp'], dtype='Int64')
    return moments_df, ball_df, events_df

def save_game_parquet(moments_df, ball_df, events_df, path):
    for df, tag in ((moments_df, 'moment'), (ball_df, 'ball'), (events_df, 'event')):
        df['record_type'] = tag
    tables = [pa.Table.from_pandas(df, preserve_index=False)
              for df in (moments_df, ball_df, events_df)]
    pq.write_table(pa.concat_tables(tables, promote_options='default'),
               path, compression='snappy')

def run_raw_to_local_parquet_conversion():
    files = get_file_list_from_github()
    pbp, has_pbp = get_event_data_from_github()
    
    errors  = []
    skipped = 0

    for f in tqdm(files, desc='Converting games'):
        stem = f['stem']
        out_path = os.path.join(OUTPUT_DIR, f'{stem}.parquet')
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            data = download_and_extract_json(f['download_url'], f['name'])
            moments_df, ball_df, events_df = parse_game(
                data, stem=stem, pbp=pbp if has_pbp else None
            )
            save_game_parquet(moments_df, ball_df, events_df, out_path)
        except Exception as e:
            errors.append({'file': f['name'], 'error': str(e)})
            print(f'\n  Error on {f['name']}: {e}')

    print(f'\nDone! Skipped {skipped} already-converted files.')
    if errors:
        print(f'{len(errors)} errors:')
        for e in errors:
            print(f'  {e['file']}: {e['error']}')
    else:
        print('No errors.')


# =============================================
# Part 2: Sample Output
# =============================================
def print_sample_parquet():
    '''
    Moments contains the literal tracking data
    Events contains descriptions of events that occurred (like play-by-play description)
    Ball is similar to Moments, but ball specific tracking data
    '''

    files = get_file_list_from_github()

    sample_stem = sorted(
        f['stem'] for f in files
        if os.path.exists(os.path.join(OUTPUT_DIR, f['stem'] + '.parquet'))
    )[0]

    game_path = os.path.join(OUTPUT_DIR, f'{sample_stem}.parquet')
    mom  = pq.read_table(game_path, filters=[('record_type', '=', 'moment')]).to_pandas()
    ball = pq.read_table(game_path, filters=[('record_type', '=', 'ball')]).to_pandas()
    evt  = pq.read_table(game_path, filters=[('record_type', '=', 'event')]).to_pandas()

    print(f'Game: {sample_stem}')
    print(f'\nmoments  {mom.shape}  |  {mom.memory_usage(deep=True).sum()/1e6:.1f} MB')
    print(f'ball     {ball.shape}')
    print(f'events   {evt.shape}')
    print('\n-- moments sample --')
    print(mom.head(10).to_string())
    print('\n-- ball sample --')
    print(ball.head(10).to_string())
    print('\n-- events sample --')
    print(evt.head(10).to_string())

    print(evt['possession_team_id'][:10].astype(str))


# =============================================
# Part 3: Re-process Shooting CSV with NBA IDs
# =============================================
def normalize_name(name):
    # First handle standard decomposable diacritics (é→e, ć→c, etc.)
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')

    # Then explicitly replace characters that don't decompose via NFD
    replacements = {
        'ş': 's', 'Ş': 'S',
        'ı': 'i', 'İ': 'I',
        'ß': 'ss',
        'ğ': 'g', 'Ğ': 'G',
        'ø': 'o', 'Ø': 'O',
        'æ': 'ae', 'Æ': 'AE',
        'đ': 'd', 'Đ': 'D',
    }
    for char, replacement in replacements.items():
        name = name.replace(char, replacement)

    # Remove periods and collapse spaces
    name = re.sub(r'\.', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def convert_height_to_in(x):
    ft, inches = x.split('-')
    total_inches = int(ft)*12 + int(inches)
    return total_inches

def get_id(x, name_to_id):
    normalized = normalize_name(x)
    if normalized in name_to_id:
        return name_to_id[normalized]

    # Fallback: try matching by stripping generational suffixes from map keys
    suffixes = r'\s+(Jr|Sr|II|III|IV)$'
    stripped = re.sub(suffixes, '', normalized, flags=re.IGNORECASE)
    for key in name_to_id:
        if re.sub(suffixes, '', key, flags=re.IGNORECASE) == stripped:
            return name_to_id[key]

    return pd.NA

def reprocess_shooting_csv():
    # Fetch all players with historical data included
    players_data = commonallplayers.CommonAllPlayers(
        is_only_current_season=0,
        league_id='00',
        season='2015-16'
    ).get_data_frames()[0]

    # Filter for players whose career spans across the 2015-16 season
    # FROM_YEAR is the start year (e.g., 2015) and TO_YEAR is the end year (e.g., 2016 or later)
    season_players = players_data[
        (players_data['FROM_YEAR'].astype(int) <= 2015) &
        (players_data['TO_YEAR'].astype(int) >= 2015)
    ]

    name_to_id = {
        normalize_name(row.DISPLAY_FIRST_LAST): int(row.PERSON_ID)
        for row in season_players.itertuples()
    }

    # Process the player shooting data
    shooting_df = pd.read_csv(SHOOTING_DATA_FILEPATH)
    shooting_df = shooting_df[shooting_df['Year'] == 2016][['Player', 'Ht', '2P%', 'FT%', '3P%']].copy()

    shooting_df['Ht'] = shooting_df['Ht'].apply(convert_height_to_in)
    shooting_df['player_id'] = shooting_df['Player'].apply(lambda x: get_id(x, name_to_id))

    shooting_df.columns = ['Player', 'height', 'two_pt_pct', 'ft_pct', 'three_pt_pct', 'player_id']

    shooting_df.to_csv(FILTERED_SHOOTING_FILEPATH, index=False)


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_raw_to_local_parquet_conversion()
    print_sample_parquet()
    reprocess_shooting_csv()