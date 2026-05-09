import numpy as np

# Global Variables
POSSESSION_RADIUS = 5.0
MIN_PASS_DISTANCE = 2.0
MIN_BALL_SPEED = 3.0
MIN_POSSESSION_FRAMES = 3
SUCCESS_POSSESSION_FRAMES = 5

# =======================
# HELPERS
# =======================
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_closest_player(ball_pos, players):
    dists = [(p['player_id'], distance(ball_pos, (p['x'], p['y']))) for p in players]
    return min(dists, key=lambda x: x[1])

def get_player_pos(frame, player_id):
    for p in frame['players']:
        if p['player_id'] == player_id:
            return (p['x'], p['y'])
    return None

def get_velocity_vector(frames, idx, player_id, window=3):
    '''Returns 2D velocity vector for a player, or None if unavailable.'''
    if idx < window:
        return None
    p1 = get_player_pos(frames[idx - window], player_id)
    p2 = get_player_pos(frames[idx], player_id)
    if p1 is None or p2 is None:
        return None
    dt = abs(frames[idx]['time'] - frames[idx - window]['time'])
    if dt == 0:
        return None
    return (np.array(p2) - np.array(p1)) / dt


# =======================
# FEATURES
# =======================
def compute_player_velocity(frames, idx, player_id, window=3):
    if idx < window:
        return 0

    p1 = get_player_pos(frames[idx - window], player_id)
    p2 = get_player_pos(frames[idx], player_id)

    if p1 is None or p2 is None:
        return 0

    dt = abs(frames[idx]['time'] - frames[idx - window]['time'])
    return distance(p1, p2) / dt if dt > 0 else 0

def point_to_segment_distance(p, a, b):
    ap = np.array(p) - np.array(a)
    ab = np.array(b) - np.array(a)

    if np.dot(ab, ab) == 0:
        return np.linalg.norm(ap)

    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = max(0, min(1, t))
    closest = np.array(a) + t * ab

    return np.linalg.norm(np.array(p) - closest)

def count_defenders_in_lane(frame, start_pos, end_pos, passer_id):
    passer_team = next(
        p['team_id'] for p in frame['players'] if p['player_id'] == passer_id
    )

    count = 0
    for p in frame['players']:
        if p['team_id'] == passer_team:
            continue

        d = point_to_segment_distance((p['x'], p['y']), start_pos, end_pos)

        if d < 3.0:
            count += 1

    return count

def check_pass_success(frames, catch_idx, receiver_id):
    stable_count = 0

    for i in range(catch_idx, min(catch_idx + 10, len(frames))):
        frame = frames[i]

        closest_id, dist = get_closest_player(
            (frame['ball']['x'], frame['ball']['y']),
            frame['players']
        )

        if closest_id == receiver_id and dist < POSSESSION_RADIUS:
            stable_count += 1
        else:
            break

    return 1 if stable_count >= SUCCESS_POSSESSION_FRAMES else 0

def compute_pass_angle(frames, idx, passer_id, pass_start_pos, pass_end_pos):
    '''
    Angle (degrees) between the passer's movement direction and the pass direction.
    0 = passing straight ahead, 90 = passing sideways, 180 = passing backward.
    Returns -1 if velocity cannot be computed.
    '''
    vel_vec = get_velocity_vector(frames, idx, passer_id, window=3)
    if vel_vec is None:
        return -1.0

    vel_norm = np.linalg.norm(vel_vec)
    if vel_norm == 0:
        return -1.0

    pass_vec = np.array(pass_end_pos) - np.array(pass_start_pos)
    pass_norm = np.linalg.norm(pass_vec)
    if pass_norm == 0:
        return -1.0

    cos_angle = np.dot(vel_vec, pass_vec) / (vel_norm * pass_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def nearest_defender_to_trajectory(frame, pass_start_pos, pass_end_pos, passer_id):
    '''
    Minimum distance from any defender to the pass trajectory (line segment).
    '''
    passer_team = next(
        p['team_id'] for p in frame['players'] if p['player_id'] == passer_id
    )

    min_dist = float('inf')
    for p in frame['players']:
        if p['team_id'] == passer_team:
            continue
        d = point_to_segment_distance((p['x'], p['y']), pass_start_pos, pass_end_pos)
        if d < min_dist:
            min_dist = d

    return float(min_dist) if min_dist != float('inf') else -1.0

def receiver_nearest_defender_dist(frame, receiver_pos, passer_team):
    '''Distance from the receiver's position to the nearest defender.'''
    dists = [
        distance(receiver_pos, (p['x'], p['y']))
        for p in frame['players']
        if p['team_id'] != passer_team
    ]
    return float(min(dists)) if dists else -1.0

def receiver_defender_closing_speed(frame, frames, frame_idx, receiver_pos, passer_team):
    '''
    Velocity component of the nearest defender directed toward the receiver.
    Positive = closing in, negative = moving away.
    Returns -1 if unavailable.
    '''
    defenders = [p for p in frame['players'] if p['team_id'] != passer_team]
    if not defenders:
        return -1.0

    nearest = min(defenders, key=lambda p: distance(receiver_pos, (p['x'], p['y'])))
    def_vel = get_velocity_vector(frames, frame_idx, nearest['player_id'], window=3)
    if def_vel is None:
        return -1.0

    to_receiver = np.array(receiver_pos) - np.array((nearest['x'], nearest['y']))
    norm = np.linalg.norm(to_receiver)
    if norm == 0:
        return -1.0

    return float(np.dot(def_vel, to_receiver / norm))

def offensive_spacing(frame, passer_team, passer_id):
    '''
    Mean pairwise distance between all offensive players excluding the passer.
    Captures how spread out the offense is at pass time.
    '''
    teammates = [
        p for p in frame['players']
        if p['team_id'] == passer_team and p['player_id'] != passer_id
    ]
    if len(teammates) < 2:
        return -1.0

    positions = [(p['x'], p['y']) for p in teammates]
    dists = [
        distance(positions[i], positions[j])
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
    ]
    return float(np.mean(dists))

def passer_nearest_defender_dist(frame, passer_id, passer_team):
    '''Distance from the passer to their nearest defender at release.'''
    passer_pos = next(
        ((p['x'], p['y']) for p in frame['players'] if p['player_id'] == passer_id),
        None
    )
    if passer_pos is None:
        return -1.0

    dists = [
        distance(passer_pos, (p['x'], p['y']))
        for p in frame['players']
        if p['team_id'] != passer_team
    ]
    return float(min(dists)) if dists else -1.0

def receiver_separation_ratio(recv_defender_dist, pass_dist):
    '''
    Receiver's defender distance normalized by pass distance.
    A 3ft cushion on a 5ft pass is very different from a 3ft cushion on a 40ft pass.
    Returns -1 if unavailable.
    '''
    if recv_defender_dist < 0 or pass_dist <= 0:
        return -1.0
    return float(recv_defender_dist / pass_dist)

def max_defender_lane_depth(frame, pass_start_pos, pass_end_pos, passer_id):
    '''
    For defenders within 3ft of the pass lane, returns the maximum projection
    depth along the pass trajectory (0 = at passer, 1 = at receiver).
    Captures whether defenders are blocking the far end vs. the near end.
    Returns -1 if no defenders are in the lane.
    '''
    passer_team = next(
        p['team_id'] for p in frame['players'] if p['player_id'] == passer_id
    )

    ab = np.array(pass_end_pos) - np.array(pass_start_pos)
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq == 0:
        return -1.0

    max_depth = -1.0
    for p in frame['players']:
        if p['team_id'] == passer_team:
            continue

        ap = np.array((p['x'], p['y'])) - np.array(pass_start_pos)
        lane_dist = point_to_segment_distance((p['x'], p['y']), pass_start_pos, pass_end_pos)

        if lane_dist < 3.0:
            t = float(np.dot(ap, ab) / ab_len_sq)
            t = max(0.0, min(1.0, t))
            if t > max_depth:
                max_depth = t

    return max_depth

def pass_trajectory_crowding(frame, pass_start_pos, pass_end_pos, radius=4.0):
    '''
    Count of ALL players (both teams) within radius of the pass lane.
    Captures general traffic independent of team.
    '''
    count = 0
    for p in frame['players']:
        d = point_to_segment_distance((p['x'], p['y']), pass_start_pos, pass_end_pos)
        if d < radius:
            count += 1
    return count

# =======================
# DATASET BUILDER
# =======================
def build_pass_dataset(frames, passes, game_id, start_pass_id):

    dataset = []

    for idx, p in enumerate(passes):
        start = p['start_frame']
        end   = p['end_frame']

        start_frame = frames[start]
        end_frame   = frames[end]

        passer_id   = p['from']
        receiver_id = p['to']

        passer_team = next(
            pl['team_id'] for pl in start_frame['players']
            if pl['player_id'] == passer_id
        )

        pass_dist = distance(p['start_pos'], p['end_pos'])

        pass_time = abs(start_frame['time'] - end_frame['time'])
        pass_speed = pass_dist / pass_time if pass_time > 0 else 0

        passer_vel   = compute_player_velocity(frames, start, passer_id)
        receiver_vel = compute_player_velocity(frames, end, receiver_id)

        defenders = count_defenders_in_lane(
            start_frame, p['start_pos'], p['end_pos'], passer_id
        )

        success = check_pass_success(frames, end, receiver_id)

        quarter        = p['quarter']
        time_remaining = p['time']
        game_time = (
            (4 - quarter) * 720 + time_remaining if quarter <= 4
            else (quarter - 5) * 300 + time_remaining
        )

        pass_angle = compute_pass_angle(
            frames, start, passer_id, p['start_pos'], p['end_pos']
        )
        nearest_def_traj = nearest_defender_to_trajectory(
            start_frame, p['start_pos'], p['end_pos'], passer_id
        )

        recv_pos             = p['end_pos']
        recv_nearest_def     = receiver_nearest_defender_dist(end_frame, recv_pos, passer_team)
        recv_closing_spd     = receiver_defender_closing_speed(end_frame, frames, end, recv_pos, passer_team)
        off_spacing          = offensive_spacing(start_frame, passer_team, passer_id)
        passer_pressure      = passer_nearest_defender_dist(start_frame, passer_id, passer_team)
        recv_sep_ratio       = receiver_separation_ratio(recv_nearest_def, pass_dist)
        max_lane_depth       = max_defender_lane_depth(start_frame, p['start_pos'], p['end_pos'], passer_id)
        traj_crowding        = pass_trajectory_crowding(start_frame, p['start_pos'], p['end_pos'])

        dataset.append({
            'pass_id':                      start_pass_id + idx,
            'game_id':                      game_id,

            'quarter':                      quarter,
            'time_remaining':               time_remaining,
            'game_time':                    game_time,

            'pass_distance':                pass_dist,
            'pass_speed':                   pass_speed,
            'pass_angle':                   pass_angle,
            'nearest_defender_dist':        nearest_def_traj,
            'max_defender_lane_depth':      max_lane_depth,
            'pass_trajectory_crowding':     traj_crowding,

            'passer_velocity':              passer_vel,
            'passer_nearest_defender_dist': passer_pressure,

            'receiver_velocity':            receiver_vel,
            'receiver_nearest_defender_dist': recv_nearest_def,
            'receiver_defender_closing_speed': recv_closing_spd,
            'receiver_separation_ratio':    recv_sep_ratio,

            'offensive_spacing':            off_spacing,
            'defenders_in_lane':            defenders,

            'is_synthetic':                 0,
            'success':                      success
        })

    return dataset


# =======================
# GENERATION OF SYNTHETIC NEGATIVES
# =======================
def score_difficulty(pl, start_frame, ball_pos, passer_team, frames, frame_idx):
    pos = (pl['x'], pl['y'])
    recv_def_dist = receiver_nearest_defender_dist(start_frame, pos, passer_team)
    lane_crowding = count_defenders_in_lane(start_frame, ball_pos, pos,
                                            next(p['player_id'] for p in start_frame['players']
                                                if p['team_id'] == passer_team
                                                and (p['x'], p['y']) == ball_pos),
                                            ) if False else 0  # skip for now
    # Lower defender distance = more defended = harder pass = better negative
    return recv_def_dist  # ascending = most defended first

def generate_synthetic_passes(frames, real_passes, game_id, start_pass_id, k_negatives=1):
    '''
    For each real detected pass, generate k counterfactual passes
    to other teammates at the same moment as a synthetic negative.
    '''
    dataset = []
    pass_id = start_pass_id

    for p in real_passes:
        start = p['start_frame']
        end   = p['end_frame']

        start_frame = frames[start]
        end_frame   = frames[end]

        passer_id   = p['from']
        receiver_id = p['to']

        passer_team = next(
            pl['team_id'] for pl in start_frame['players']
            if pl['player_id'] == passer_id
        )

        eligible = [
            pl for pl in start_frame['players']
            if pl['team_id'] == passer_team
            and pl['player_id'] != passer_id
            and pl['player_id'] != receiver_id
        ]

        ball_pos = p['start_pos']

        eligible_sorted = sorted(
            eligible,
            key=lambda pl: score_difficulty(pl, start_frame, ball_pos, passer_team, frames, start)
        )
        synthetic_targets = eligible_sorted[:k_negatives]

        for target in synthetic_targets:
            target_pos = (target['x'], target['y'])

            if is_likely_successful(start_frame, ball_pos, target_pos, target['player_id'], frames, start):
                continue

            synth_dist = distance(ball_pos, target_pos)
            if synth_dist < MIN_PASS_DISTANCE:
                continue

            pass_time  = abs(start_frame['time'] - end_frame['time'])
            synth_speed = synth_dist / pass_time if pass_time > 0 else 0

            passer_vel   = compute_player_velocity(frames, start, passer_id)
            receiver_vel = compute_player_velocity(frames, start, target['player_id'])

            defenders = count_defenders_in_lane(
                start_frame, ball_pos, target_pos, passer_id
            )

            quarter        = p['quarter']
            time_remaining = p['time']
            game_time = (
                (4 - quarter) * 720 + time_remaining if quarter <= 4
                else (quarter - 5) * 300 + time_remaining
            )

            pass_angle = compute_pass_angle(
                frames, start, passer_id, ball_pos, target_pos
            )
            nearest_def_traj = nearest_defender_to_trajectory(
                start_frame, ball_pos, target_pos, passer_id
            )

            # For synthetics, use start_frame for receiver context since
            # there is no real catch frame — target_pos is where they are now
            recv_nearest_def  = receiver_nearest_defender_dist(start_frame, target_pos, passer_team)
            recv_closing_spd  = receiver_defender_closing_speed(start_frame, frames, start, target_pos, passer_team)
            off_spacing       = offensive_spacing(start_frame, passer_team, passer_id)
            passer_pressure   = passer_nearest_defender_dist(start_frame, passer_id, passer_team)
            recv_sep_ratio    = receiver_separation_ratio(recv_nearest_def, synth_dist)
            max_lane_depth    = max_defender_lane_depth(start_frame, ball_pos, target_pos, passer_id)
            traj_crowding     = pass_trajectory_crowding(start_frame, ball_pos, target_pos)

            dataset.append({
                'pass_id':                      pass_id,
                'game_id':                      game_id,

                'quarter':                      quarter,
                'time_remaining':               time_remaining,
                'game_time':                    game_time,

                'pass_distance':                synth_dist,
                'pass_speed':                   synth_speed,
                'pass_angle':                   pass_angle,
                'nearest_defender_dist':        nearest_def_traj,
                'max_defender_lane_depth':      max_lane_depth,
                'pass_trajectory_crowding':     traj_crowding,

                'passer_velocity':              passer_vel,
                'passer_nearest_defender_dist': passer_pressure,

                'receiver_velocity':            receiver_vel,
                'receiver_nearest_defender_dist': recv_nearest_def,
                'receiver_defender_closing_speed': recv_closing_spd,
                'receiver_separation_ratio':    recv_sep_ratio,

                'offensive_spacing':            off_spacing,
                'defenders_in_lane':            defenders,

                'is_synthetic':                 1,
                'success':                      0
            })
            pass_id += 1

    return dataset, pass_id

def is_likely_successful(start_frame, ball_pos, target_pos, target_id, frames, frame_idx):
    '''
    Returns True if a hypothetical pass to target looks like it would succeed.
    These should be excluded from synthetic negatives.
    '''
    passer_id = None
    passer_team = None
    for pl in start_frame['players']:
        if (pl['x'], pl['y']) == ball_pos:
            passer_id = pl['player_id']
            passer_team = pl['team_id']
            break

    if passer_id is None:
        return False

    defenders_in_lane = count_defenders_in_lane(
        start_frame, ball_pos, target_pos, passer_id
    )
    if defenders_in_lane == 0:

        target_pos_arr = np.array(target_pos)
        defender_dists = [
            distance(target_pos, (pl['x'], pl['y']))
            for pl in start_frame['players']
            if pl['team_id'] != passer_team
        ]
        receiver_is_open = min(defender_dists) > 4.0 if defender_dists else True

        pass_dist    = distance(ball_pos, target_pos)
        is_short     = pass_dist < 10.0

        receiver_vel_vec = get_velocity_vector(frames, frame_idx, target_id, window=3)
        pass_direction   = target_pos_arr - np.array(ball_pos)
        norm             = np.linalg.norm(pass_direction)
        moving_toward    = False
        if norm > 0 and receiver_vel_vec is not None:
            moving_toward = np.dot(receiver_vel_vec, pass_direction / norm) > 0

        if receiver_is_open and (is_short or moving_toward):
            return True

    return False
