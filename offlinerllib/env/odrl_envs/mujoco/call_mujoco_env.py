from typing import Dict
from pathlib import Path
import gym

from gym.envs.mujoco.half_cheetah_v3    import  HalfCheetahEnv
from gym.envs.mujoco.ant_v3             import  AntEnv
from gym.envs.mujoco.walker2d_v3        import  Walker2dEnv
from gym.envs.mujoco.hopper_v3          import  HopperEnv

from gym.wrappers.time_limit            import  TimeLimit


def call_mujoco_env(env_config: Dict) -> gym.Env:
    env_name     =   env_config['env_name'].lower()     #   eg. "hopper_friction" or "hopper_morph_foot", body_pard is required only in "morph" or 'kinematic" mode
    shift_level  =   env_config['shift_level']          #   either float(0.1/0.5/...) or level(easy/medium/hard)

    if '-' in env_name:
        env_name = env_name.replace('-', '_')

    # assert the shift level legal
    if 'morph' in env_name or 'kinematic' in env_name:
        assert shift_level in ['easy', 'medium', 'hard'], 'The required shift is not available yet, please consider modify the xml file on your own or use the shift scale among easy, medium, hard'
    if 'friction' in env_name or 'gravity' in env_name:
        assert float(shift_level) in [0.1, 0.5, 1.0, 2.0, 5.0], 'The required shift is not available yet, please consider modify the xml file on your own or use the shift scale among 0.1, 0.5, 2.0, 5.0'

    # decide which task it is, support the following tasks
    # hopper/half_cheetah/walker2d/ant - friction
    #                                  - gravity
    #                                  - morph
    #                                  - noise
    #                                  - broken

    if 'hopper' in env_name:
        if env_name == 'hopper':
            return gym.make('Hopper-v3')
        elif 'friction' in env_name or 'gravity' in env_name:
            if float(shift_level) == 1.0:
                return gym.make('Hopper-v3')
            return TimeLimit(
                HopperEnv(xml_file=f"{str(Path(__file__).parent.absolute())}/assets/{env_name}_{float(shift_level)}.xml",),
                max_episode_steps=1000
            )
        elif 'noise' in env_name:
            # todo: the modification is directly applied on the executed action, no need to modify the xml file itself
            return gym.make('Hopper-v3')
        elif 'morph' in env_name or 'kinematic' in env_name:
            return TimeLimit(
                HopperEnv(xml_file=f"{str(Path(__file__).parent.absolute())}/assets/{env_name}_{shift_level}.xml",),
                max_episode_steps=1000          
            )
        else:
            print("env_name {env_name} is illegal or not implemented")
            raise NotImplementedError
    elif "halfcheetah" in env_name:
        if env_name == 'halfcheetah':
            return gym.make('HalfCheetah-v3')
        elif 'friction' in env_name or 'gravity' in env_name:
            if float(shift_level) == 1.0:
                return gym.make('Hopper-v3')
            return TimeLimit(
                HalfCheetahEnv(xml_file=f"{str(Path(__file__).parent.absolute())}/assets/{env_name}_{float(shift_level)}.xml",),
                max_episode_steps=1000            
            )
        elif 'noise' in env_name:
            # todo: the modification is directly applied on the executed action, no need to modify the xml file itself
            return gym.make('HalfCheetah-v3')
        elif 'morph' in env_name or 'kinematic' in env_name:
            return TimeLimit(
                HalfCheetahEnv(xml_file=f"{str(Path(__file__).parent.absolute())}/assets/{env_name}_{shift_level}.xml",),
                max_episode_steps=1000          
            )
        else:
            print("env_name {env_name} is illegal or not implemented")
            raise NotImplementedError
    elif "walker2d" in env_name:
        if env_name == 'walker2d':
            return gym.make('Walker2d-v3')
        elif 'friction' in env_name or 'gravity' in env_name:
            if float(shift_level) == 1.0:
                return gym.make('Hopper-v3')
            return TimeLimit(
                Walker2dEnv(xml_file=f"{str(Path(__file__).parent.absolute())}/assets/{env_name}_{float(shift_level)}.xml",),
                max_episode_steps=1000            
            )
        elif 'noise' in env_name:
            # todo: the modification is directly applied on the executed action, no need to modify the xml file itself
            return gym.make('Walker2d-v3')
        elif 'morph' in env_name or 'kinematic' in env_name:
            return TimeLimit(
                Walker2dEnv(xml_file=f"{str(Path(__file__).parent.absolute())}/assets/{env_name}_{shift_level}.xml",),
                max_episode_steps=1000          
            )
        else:
            print("env_name {env_name} is illegal or not implemented")
            raise NotImplementedError
    elif 'ant' in env_name:
        if env_name == 'ant':
            return gym.make('Ant-v3')
        elif 'friction' in env_name or 'gravity' in env_name:
            if float(shift_level) == 1.0:
                return gym.make('Hopper-v3')
            return TimeLimit(
                AntEnv(xml_file=f"{str(Path(__file__).parent.absolute())}/assets/{env_name}_{float(shift_level)}.xml",),
                max_episode_steps=1000            
            )
        elif 'noise' in env_name:
            # todo: the modification is directly applied on the executed action, no need to modify the xml file itself
            return gym.make('Ant-v3')
        elif 'morph' in env_name or 'kinematic' in env_name:
            return TimeLimit(
                AntEnv(xml_file=f"{str(Path(__file__).parent.absolute())}/assets/{env_name}_{shift_level}.xml",),
                max_episode_steps=1000          
            )
        else:
            print("env_name {env_name} is illegal or not implemented")
            raise NotImplementedError
    else:
        print("env_name {env_name} is illegal or not implemented")
        raise NotImplementedError