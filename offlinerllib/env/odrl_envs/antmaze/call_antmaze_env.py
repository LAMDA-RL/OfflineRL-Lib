from typing import Dict
import gym
import d4rl


def call_antmaze_env(env_config: Dict) -> gym.Env:
    env_name     =   env_config['env_name'].lower()     #   eg. "antmaze_small_lshape"
    if '_' in env_name:
        env_name = env_name.replace('_', '-')
    # decide which task it is, support the following tasks
    # antmaze       -   small          - empty
    #                                  - lshape
    #                                  - centerblock
    #                                  - brokenjoint
    #                                  - reversel
    #                                  - reverseu
    #                                  - zshape
    #               -   medium         - 1/2/3/4/5/6
    #               -   large          - 1/2/3/4/5/6
    assert env_name.startswith('antmaze')
    assert any([size in env_name for size in ['small', 'medium', 'large']])

    shift_level = env_config['shift_level']

    if shift_level is None:
        if 'small' in env_name:
            return gym.make('antmaze-umaze-v0')
        elif 'medium' in env_name:
            return gym.make('antmaze-medium-0-v0')
        else:
            return gym.make('antmaze-large-0-v0')
    else:
        if 'small' in env_name:
            assert any([size in shift_level for size in ['empty', 'lshape', 'centerblock', 'reversel', 'reverseu', 'zshape']])
            env_name += '-' + str(shift_level) + '-v0'
        elif 'medium' in env_name:
            assert any([size in shift_level for size in ['0','1', '2', '3', '4', '5', '6']])
            env_name += '-' + str(shift_level) + '-v0'
        else:
            assert any([size in shift_level for size in ['1', '2', '3', '4', '5', '6']])
            env_name += '-' + str(shift_level) + '-v0'
        return gym.make(env_name)