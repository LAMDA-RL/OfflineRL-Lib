import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import random
from UtilsRL.exp import parse_args

args = parse_args()
if args.env_type == "dmc":
    args.env = "-".join([args.domain.title(), args.task.title(), "v1"])
elif args.env_type == "mujoco":
    args.env = args.task
elif args.env_type == "robosuite":
    args.env = args.task
    args.robots = args.robots
data_path = f"./datasets/rpl/{args.name}/{args.env}/data.npz"

# Load data file
data = np.load(data_path, allow_pickle=True)
print("Loaded data shapes:")
for k, v in data.items():
    print(k, v.shape)

# Split data into datasets
preference_train_data = {k: v[:2700] for k, v in data.items()}
preference_eval_data = {k: v[2700:] for k, v in data.items()}

# Save offline datasets
saved_path = f"./datasets/rpl/{args.name}/{args.env}/"
os.makedirs(saved_path, exist_ok=True)

# Function to generate preference data
def generate_preference_data(data, num_samples, segment_len, discount=0.99):
    obs_1 = []
    obs_2 = []
    action_1 = []
    action_2 = []
    next_obs_1 = []
    next_obs_2 = []
    reward_1 = []
    reward_2 = []
    timestep_1 = []
    timestep_2 = []
    terminal_1 = []
    terminal_2 = []
    rl_dir_label = []
    rl_dis_dir_label = []
    rl_sum_label = []
    rl_dis_sum_label = []
    rl_dir_1 = []
    rl_dir_2 = []
    rl_dis_dir_1 = []
    rl_dis_dir_2 = []
    rl_sum_1 = []
    rl_sum_2 = []
    rl_dis_sum_1 = []
    rl_dis_sum_2 = []
    q_value_1 = []
    q_value_2 = []
    v_value_1 = []
    v_value_2 = []
    next_v_value_1 = []
    next_v_value_2 = []

    num_episodes = data['obs'].shape[0]
    max_timesteps = data['obs'].shape[1] - segment_len

    # Determine trajectory lengths using data['mask']
    trajectory_lengths = np.sum(data['mask'], axis=1).astype(int)
    
    # V1: Precompute all valid segment indices where masks are all 1
    # segment_indices = [
    #     (ep_idx, seg_idx * segment_len, (seg_idx + 1) * segment_len)
    #     for ep_idx in range(data['obs'].shape[0])
    #     for seg_idx in range(trajectory_lengths[ep_idx] // segment_len)
    # ]
    
    for _ in range(num_samples):

        # V1: Use precomputed segment_indices to select segments
        # ep_idx1, t_start1, t_end1 = random.choice(segment_indices)
        # ep_idx2, t_start2, t_end2 = random.choice(segment_indices)

        # V2: random start
        ep_idx1 = random.randint(0, data['obs'].shape[0] - 1)
        ep_idx2 = random.randint(0, data['obs'].shape[0] - 1)
        t_start1 = random.randint(0, trajectory_lengths[ep_idx1] - segment_len)
        t_start2 = random.randint(0, trajectory_lengths[ep_idx2] - segment_len)
        t_end1 = t_start1 + segment_len
        t_end2 = t_start2 + segment_len
        
        # Ensure masks are all 1
        assert np.all(data['mask'][ep_idx1, t_start1:t_end1] == 1)
        assert np.all(data['mask'][ep_idx2, t_start2:t_end2] == 1)
    
        t_idx1 = t_start1
        t_idx2 = t_start2

        # Extract segments
        obs_seg_1 = data['obs'][ep_idx1, t_idx1:t_idx1+segment_len]
        obs_seg_2 = data['obs'][ep_idx2, t_idx2:t_idx2+segment_len]

        action_seg_1 = data['action'][ep_idx1, t_idx1:t_idx1+segment_len]
        action_seg_2 = data['action'][ep_idx2, t_idx2:t_idx2+segment_len]

        next_obs_seg_1 = data['next_obs'][ep_idx1, t_idx1:t_idx1+segment_len]
        next_obs_seg_2 = data['next_obs'][ep_idx2, t_idx2:t_idx2+segment_len]

        reward_seg_1 = data['reward'][ep_idx1, t_idx1:t_idx1+segment_len].squeeze()
        reward_seg_2 = data['reward'][ep_idx2, t_idx2:t_idx2+segment_len].squeeze()

        q_value_seg_1 = data['q_value'][ep_idx1, t_idx1:t_idx1+segment_len].squeeze()
        q_value_seg_2 = data['q_value'][ep_idx2, t_idx2:t_idx2+segment_len].squeeze()

        v_value_seg_1 = data['v_value'][ep_idx1, t_idx1:t_idx1+segment_len].squeeze()
        v_value_seg_2 = data['v_value'][ep_idx2, t_idx2:t_idx2+segment_len].squeeze()

        next_v_value_seg_1 = data['next_v_value'][ep_idx1, t_idx1:t_idx1+segment_len].squeeze()
        next_v_value_seg_2 = data['next_v_value'][ep_idx2, t_idx2:t_idx2+segment_len].squeeze()

        # terminal
        terminal_seg_1 = data['terminal'][ep_idx1, t_idx1:t_idx1+segment_len].astype(float).squeeze()
        terminal_seg_2 = data['terminal'][ep_idx2, t_idx2:t_idx2+segment_len].astype(float).squeeze()

        # label_keys:
        # rl_dir: \sum_{t} Q(s_t, a_t) - V(s_t)
        # rl_dis_dir: \sum_{t} \gamma^t (Q(s_t, a_t) - V(s_t))
        # rl_sum: \sum_{t} [r_t] + V(s_T) - V(s_0)
        # rl_dis_sum: \sum_{t} \sum_{t} [\gamma^t r_t] + \gamma^{T-1} V(s_T) - V(s_0)

        # Compute discounted advantage
        discounts = discount ** np.arange(segment_len)
        
        # Compute rl_dir and rl_dis_dir
        rl_dir_1_val = np.sum(q_value_seg_1 - v_value_seg_1)
        rl_dir_2_val = np.sum(q_value_seg_2 - v_value_seg_2)
        
        rl_dis_dir_1_val = np.sum((q_value_seg_1 - v_value_seg_1) * discounts)
        rl_dis_dir_2_val = np.sum((q_value_seg_2 - v_value_seg_2) * discounts)
        
        # Compute rl_sum and rl_dis_sum
        rl_sum_1_val = np.sum(reward_seg_1) + (1-terminal_seg_1[-1]) * next_v_value_seg_1[-1] - v_value_seg_1[0]
        rl_sum_2_val = np.sum(reward_seg_2) + (1-terminal_seg_2[-1]) * next_v_value_seg_2[-1] - v_value_seg_2[0]

        rl_dis_sum_1_val = np.sum(reward_seg_1 * discounts) + (1-terminal_seg_1[-1]) * (discount ** segment_len) * next_v_value_seg_1[-1] - v_value_seg_1[0]
        rl_dis_sum_2_val = np.sum(reward_seg_2 * discounts) + (1-terminal_seg_2[-1]) * (discount ** segment_len) * next_v_value_seg_2[-1] - v_value_seg_2[0]

        # Assign labels based on advantage comparisons
        rl_dir_label.append(0. if rl_dir_1_val > rl_dir_2_val else 1.)
        rl_dis_dir_label.append(0. if rl_dis_dir_1_val > rl_dis_dir_2_val else 1.)
        rl_sum_label.append(0. if rl_sum_1_val > rl_sum_2_val else 1.)
        rl_dis_sum_label.append(0. if rl_dis_sum_1_val > rl_dis_sum_2_val else 1.)

        obs_1.append(obs_seg_1)
        obs_2.append(obs_seg_2)
        action_1.append(action_seg_1)
        action_2.append(action_seg_2)
        next_obs_1.append(next_obs_seg_1)
        next_obs_2.append(next_obs_seg_2)
        reward_1.append(reward_seg_1)
        reward_2.append(reward_seg_2)
        timestep_1.append(np.arange(t_idx1, t_idx1+segment_len))
        timestep_2.append(np.arange(t_idx2, t_idx2+segment_len))
        terminal_1.append(terminal_seg_1)
        terminal_2.append(terminal_seg_2)
        rl_dir_1.append(rl_dir_1_val)
        rl_dir_2.append(rl_dir_2_val)
        rl_dis_dir_1.append(rl_dis_dir_1_val)
        rl_dis_dir_2.append(rl_dis_dir_2_val)
        rl_sum_1.append(rl_sum_1_val)
        rl_sum_2.append(rl_sum_2_val)
        rl_dis_sum_1.append(rl_dis_sum_1_val)
        rl_dis_sum_2.append(rl_dis_sum_2_val)
        # Append additional variables
        q_value_1.append(q_value_seg_1)
        q_value_2.append(q_value_seg_2)
        v_value_1.append(v_value_seg_1)
        v_value_2.append(v_value_seg_2)
        next_v_value_1.append(next_v_value_seg_1)
        next_v_value_2.append(next_v_value_seg_2)

    # Create preference data dictionary
    preference_data = {
        'obs_1': np.array(obs_1),
        'obs_2': np.array(obs_2),
        'action_1': np.array(action_1),
        'action_2': np.array(action_2),
        'next_obs_1': np.array(next_obs_1),
        'next_obs_2': np.array(next_obs_2),
        'reward_1': np.array(reward_1),
        'reward_2': np.array(reward_2),
        'timestep_1': np.array(timestep_1),
        'timestep_2': np.array(timestep_2),
        'terminal_1': np.array(terminal_1),
        'terminal_2': np.array(terminal_2),
        'q_value_1': np.array(q_value_1, dtype=float),
        'q_value_2': np.array(q_value_2, dtype=float),
        'v_value_1': np.array(v_value_1, dtype=float),
        'v_value_2': np.array(v_value_2, dtype=float),
        'next_v_value_1': np.array(next_v_value_1, dtype=float),
        'next_v_value_2': np.array(next_v_value_2, dtype=float),
        'rl_dir_label': np.array(rl_dir_label, dtype=float),
        'rl_dis_dir_label': np.array(rl_dis_dir_label, dtype=float),
        'rl_sum_label': np.array(rl_sum_label, dtype=float),
        'rl_dis_sum_label': np.array(rl_dis_sum_label, dtype=float),
        'rl_dir_1': np.array(rl_dir_1, dtype=float),
        'rl_dir_2': np.array(rl_dir_2, dtype=float),
        'rl_dis_dir_1': np.array(rl_dis_dir_1, dtype=float),
        'rl_dis_dir_2': np.array(rl_dis_dir_2, dtype=float),
        'rl_sum_1': np.array(rl_sum_1, dtype=float).squeeze(),
        'rl_sum_2': np.array(rl_sum_2, dtype=float).squeeze(),
        'rl_dis_sum_1': np.array(rl_dis_sum_1, dtype=float).squeeze(),
        'rl_dis_sum_2': np.array(rl_dis_sum_2, dtype=float).squeeze(),
    }
    return preference_data

# Generate preference datasets using batch processing
segment_len = 64
num_samples_train = 20000
num_samples_eval = 1000
batch_size_train = 1350
batch_size_eval = 1350

# Function to process and save preference data in batches
def process_and_save(preference_data, num_samples, segment_len, discount, batch_size, save_prefix):
    total = preference_data['obs'].shape[0]
    # Initialize lists to accumulate batch data
    accumulated_data = {}
    num_batch = len(list(range(0, total, batch_size)))
    for start in range(0, total, batch_size):
        batch_data = {k: v[start:start+batch_size] for k, v in preference_data.items()}
        preference_batch = generate_preference_data(
            batch_data,
            num_samples // num_batch,
            segment_len,
            discount=discount
        )
        # Accumulate batch data
        for key in preference_batch:
            if key not in accumulated_data:
                accumulated_data[key] = []
            accumulated_data[key].append(preference_batch[key])
    # Concatenate all batches
    for key in accumulated_data:
        accumulated_data[key] = np.concatenate(accumulated_data[key], axis=0)
    # Save accumulated preference data to a single file
    np.savez_compressed(saved_path + f"{save_prefix}_data.npz", **accumulated_data)
    
# random seed
np.random.seed(42)

# Generate and save preference data for training
process_and_save(
    preference_train_data,
    num_samples_train,
    segment_len,
    discount=args.discount,
    batch_size=batch_size_train,
    save_prefix="preference_train"
)

# Generate and save preference data for evaluation
process_and_save(
    preference_eval_data,
    num_samples_eval,
    segment_len,
    discount=args.discount,
    batch_size=batch_size_eval,
    save_prefix="preference_eval"
)
