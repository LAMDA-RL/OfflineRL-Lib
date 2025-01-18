from typing import Dict
import gym


def call_adroit_env(env_config: Dict) -> gym.Env:
    env_name     =   env_config['env_name'].lower()     #   eg. "pen_shrink_finger"
    shift_level  =   env_config['shift_level']          #   level(easy/medium/hard)

    if '_' in env_name:
        env_name = env_name.replace('_', '-')
    # decide which task it is, support the following tasks
    # pen/hammer/relocate/door         - shrink_finger
    #                                  - broken_joint
    assert any([env_name.startswith(f'{e}') for e in ['pen', 'hammer', 'relocate', 'door']])
    assert any([env_name.endswith(f'{e}') for e in ['shrink-finger', 'broken-joint']])

    env_name = env_name + '-' + str(shift_level) + '-v0'

    return gym.make(env_name)