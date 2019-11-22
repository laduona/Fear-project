from gym.envs.registration import register
import gym


# delete if it's registered
env_name = 'Gridworld-v0'
if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]

# register the environment
gym.register(
    id=env_name,
    entry_point='gym_gridworld.envs:GridworldEnv')