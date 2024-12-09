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
offline_train_data = {k: v[:2000] for k, v in data.items()}
offline_eval_data = {k: v[2000:2200] for k, v in data.items()}
preference_train_data = {k: v[2200:2900] for k, v in data.items()}
preference_eval_data = {k: v[2900:] for k, v in data.items()}

# Save offline datasets
saved_path = f"./datasets/rpl/{args.name}/{args.env}/"
os.makedirs(saved_path, exist_ok=True)
np.savez_compressed(saved_path + "offline_train_data.npz", **offline_train_data)
np.savez_compressed(saved_path + "offline_eval_data.npz", **offline_eval_data)

# Function to generate preference data
def generate_preference_data(data, num_samples, segment_len, discount=0.99, advantage_method='QV'):
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
    label = []

    num_episodes = data['obs'].shape[0]
    max_timesteps = data['obs'].shape[1] - segment_len

    for _ in range(num_samples):
        # Randomly select two episodes and starting timesteps
        ep_idx1, ep_idx2 = np.random.choice(num_episodes, size=2, replace=True)
        t_idx1 = np.random.randint(0, max_timesteps)
        t_idx2 = np.random.randint(0, max_timesteps - segment_len)

        # Extract segments
        obs_seg_1 = data['obs'][ep_idx1, t_idx1:t_idx1+segment_len, :11]
        obs_seg_2 = data['obs'][ep_idx2, t_idx2:t_idx2+segment_len, :11]

        action_seg_1 = data['action'][ep_idx1, t_idx1:t_idx1+segment_len, :3]
        action_seg_2 = data['action'][ep_idx2, t_idx2:t_idx2+segment_len, :3]

        next_obs_seg_1 = data['next_obs'][ep_idx1, t_idx1:t_idx1+segment_len, :11]
        next_obs_seg_2 = data['next_obs'][ep_idx2, t_idx2:t_idx2+segment_len, :11]

        reward_seg_1 = data['reward'][ep_idx1, t_idx1:t_idx1+segment_len, 0]
        reward_seg_2 = data['reward'][ep_idx2, t_idx2:t_idx2+segment_len, 0]

        q_value_seg_1 = data['q_value'][ep_idx1, t_idx1:t_idx1+segment_len, 0]
        q_value_seg_2 = data['q_value'][ep_idx2, t_idx2:t_idx2+segment_len, 0]

        v_value_seg_1 = data['v_value'][ep_idx1, t_idx1:t_idx1+segment_len, 0]
        v_value_seg_2 = data['v_value'][ep_idx2, t_idx2:t_idx2+segment_len, 0]

        # Extract next_v_value segments if using 'VV' method
        if advantage_method == 'VV':
            next_v_value_seg_1 = data['next_v_value'][ep_idx1, t_idx1:t_idx1+segment_len, 0]
            next_v_value_seg_2 = data['next_v_value'][ep_idx2, t_idx2:t_idx2+segment_len, 0]

        # Compute discounted advantage
        discounts = discount ** np.arange(segment_len)
        
        if advantage_method == 'QV':
            advantage_1 = q_value_seg_1 - v_value_seg_1
            advantage_2 = q_value_seg_2 - v_value_seg_2
        elif advantage_method == 'VV':
            advantage_1 = reward_seg_1 + discount * next_v_value_seg_1 - v_value_seg_1
            advantage_2 = reward_seg_2 + discount * next_v_value_seg_2 - v_value_seg_2
        else:
            raise ValueError("Invalid advantage_method. Choose 'QV' or 'VV'.")

        sum_advantage_1 = np.sum(advantage_1 * discounts)
        sum_advantage_2 = np.sum(advantage_2 * discounts)

        # Determine label
        if sum_advantage_1 > sum_advantage_2:
            label.append(0.)  # Segment 1 is better
        else:
            label.append(1.)  # Segment 2 is better

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
        'advantage_1': np.array(advantage_1),
        'advantage_2': np.array(advantage_2),
        'timestep_1': np.array(timestep_1),
        'timestep_2': np.array(timestep_2),
        'label': np.array(label),
    }
    return preference_data

# Generate preference datasets using both 'QV' and 'VV' advantage methods
segment_len = 64
num_samples_train = 500
num_samples_eval = 50

# Generate preference data using 'QV' method
preference_train_qv = generate_preference_data(
    preference_train_data,
    num_samples_train,
    segment_len,
    discount=args.discount,
    advantage_method='QV'
)
preference_eval_qv = generate_preference_data(
    preference_eval_data,
    num_samples_eval,
    segment_len,
    discount=args.discount,
    advantage_method='QV'
)

# Generate preference data using 'VV' method
preference_train_vv = generate_preference_data(
    preference_train_data,
    num_samples_train,
    segment_len,
    discount=args.discount,
    advantage_method='VV'
)
preference_eval_vv = generate_preference_data(
    preference_eval_data,
    num_samples_eval,
    segment_len,
    discount=args.discount,
    advantage_method='VV'
)

# Save preference datasets
np.savez_compressed(saved_path + "preference_adv_qv_train_data.npz", **preference_train_qv)
np.savez_compressed(saved_path + "preference_adv_qv_eval_data.npz", **preference_eval_qv)
np.savez_compressed(saved_path + "preference_adv_vv_train_data.npz", **preference_train_vv)
np.savez_compressed(saved_path + "preference_adv_vv_eval_data.npz", **preference_eval_vv)
