import gym
from gym.envs.registration import register
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.absolute()))
from ant import make_ant_maze_env
import ant

RESET = R = 'r'  # Reset position.
GOAL = G = 'g'

# small mazes
register(
    id='antmaze-small-0-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1],
            [1, R, 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, G, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)
register(
    id='antmaze-small-empty-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1],
            [1, R, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, G, 1],
            [1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)
register(
    id='antmaze-small-centerblock-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1],
            [1, R, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, G, 1],
            [1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)
register(
    id='antmaze-small-lshape-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1],
            [1, R, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 0, 0, G, 1],
            [1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)

register(
    id='antmaze-small-zshape-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1],
            [1, R, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 0, G, 1],
            [1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)

register(
    id='antmaze-small-reversel-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1],
            [1, R, 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, G, 1],
            [1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)

register(
    id='antmaze-small-reverseu-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1],
            [1, R, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, G, 1],
            [1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)

# medium mazes
register(
    id='antmaze-medium-0-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)
register(
    id='antmaze-medium-1-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)
register(
    id='antmaze-medium-2-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 0, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)
register(
    id='antmaze-medium-3-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 1],
            [1, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)

register(
    id='antmaze-medium-4-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)

register(
    id='antmaze-medium-5-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)

register(
    id='antmaze-medium-6-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)

# large mazes
register(
    id='antmaze-large-0-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)

register(
    id='antmaze-large-1-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)

register(
    id='antmaze-large-2-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)
register(
    id='antmaze-large-3-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)

register(
    id='antmaze-large-4-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)

register(
    id='antmaze-large-5-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)

register(
    id='antmaze-large-6-v0',
    entry_point='ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ],
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)