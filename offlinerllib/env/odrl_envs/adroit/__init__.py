from gym.envs.registration import register

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.absolute()))

# from door import DoorEnvV0
# from relocate import RelocateEnvV0
# from hammer import HammerEnvV0
# from pen import PenEnvV0
import door
import pen
import relocate
import hammer

# no need to register if d4rl is install, else please register the following environments

# register(id='door-v0', entry_point='DoorEnvV0', max_episode_steps=200, kwargs={})
# register(id='relocate-v0', entry_point='RelocateEnvV0', max_episode_steps=200, kwargs={})
# register(id='hammer-v0', entry_point='HammerEnvV0', max_episode_steps=200, kwargs={})
# register(id='pen-v0', entry_point='PenEnvV0', max_episode_steps=200, kwargs={})

register(
    id='door-shrink-finger-easy-v0', entry_point='door:DoorEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'door_shrink_finger_easy'
    }
)

register(
    id='door-shrink-finger-medium-v0', entry_point='door:DoorEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'door_shrink_finger_medium'
    }
)

register(
    id='door-shrink-finger-hard-v0', entry_point='door:DoorEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'door_shrink_finger_hard'
    }
)

register(
    id='relocate-shrink-finger-easy-v0', entry_point='relocate:RelocateEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'relocate_shrink_finger_easy'
    }
)

register(
    id='relocate-shrink-finger-medium-v0', entry_point='relocate:RelocateEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'relocate_shrink_finger_medium'
    }
)

register(
    id='relocate-shrink-finger-hard-v0', entry_point='relocate:RelocateEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'relocate_shrink_finger_hard'
    }
)

register(
    id='hammer-shrink-finger-easy-v0', entry_point='hammer:HammerEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'hammer_shrink_finger_easy'
    }
)

register(
    id='hammer-shrink-finger-medium-v0', entry_point='hammer:HammerEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'hammer_shrink_finger_medium'
    }
)

register(
    id='hammer-shrink-finger-hard-v0', entry_point='hammer:HammerEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'hammer_shrink_finger_hard'
    }
)

register(
    id='pen-shrink-finger-easy-v0', entry_point='pen:PenEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'pen_shrink_finger_easy'
    }
)

register(
    id='pen-shrink-finger-medium-v0', entry_point='pen:PenEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'pen_shrink_finger_medium'
    }
)

register(
    id='pen-shrink-finger-hard-v0', entry_point='pen:PenEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'pen_shrink_finger_hard'
    }
)

register(
    id='door-broken-joint-easy-v0', entry_point='door:DoorEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'door_broken_joint_easy'
    }
)

register(
    id='door-broken-joint-medium-v0', entry_point='door:DoorEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'door_broken_joint_medium'
    }
)

register(
    id='door-broken-joint-hard-v0', entry_point='door:DoorEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'door_broken_joint_hard'
    }
)

register(
    id='relocate-broken-joint-easy-v0', entry_point='relocate:RelocateEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'relocate_broken_joint_easy'
    }
)

register(
    id='relocate-broken-joint-medium-v0', entry_point='relocate:RelocateEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'relocate_broken_joint_medium'
    }
)

register(
    id='relocate-broken-joint-hard-v0', entry_point='relocate:RelocateEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'relocate_broken_joint_hard'
    }
)

register(
    id='hammer-broken-joint-easy-v0', entry_point='hammer:HammerEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'hammer_broken_joint_easy'
    }
)

register(
    id='hammer-broken-joint-medium-v0', entry_point='hammer:HammerEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'hammer_broken_joint_medium'
    }
)

register(
    id='hammer-broken-joint-hard-v0', entry_point='hammer:HammerEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'hammer_broken_joint_hard'
    }
)

register(
    id='pen-broken-joint-easy-v0', entry_point='pen:PenEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'pen_broken_joint_easy'
    }
)

register(
    id='pen-broken-joint-medium-v0', entry_point='pen:PenEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'pen_broken_joint_medium'
    }
)

register(
    id='pen-broken-joint-hard-v0', entry_point='pen:PenEnvV0', max_episode_steps=200, 
    kwargs={
        'xml_file': 'pen_broken_joint_hard'
    }
)