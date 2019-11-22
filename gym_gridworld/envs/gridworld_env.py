import gym
from gym.envs.classic_control import rendering
import numpy as np
import os
from gym.utils import seeding
import copy

EMPTY = WHITE = 0
WALL = BLACK = 1
TARGET = BLUE = 2
AGENT = GRAY = 3
HOPE = GREEN = 4
FEAR = RED = 5
CANDY = BROWN = 6
GHOST = SILVER =7


COLORS = {WHITE:[1.0, 1.0,1.0], BLACK:[0.0, 0.0, 0.0], BLUE:[0.0, 0.0, 1.0], GRAY:[0.5, 0.5, 0.5], GREEN:[0.0, 1.0, 0.0],
          RED:[1.0, 0.0, 0.0], BROWN:[0.5, 0.0, 0.0], SILVER:[0.75, 0.75, 0.75]}


#Action code
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class GridworldEnv(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self):

        self.target_reward = 10
        self.wall_reward = -0.1
        self.ghost_reward = -10
        self.empty_step_reward = -0.01

        self.actions = [UP, DOWN, LEFT, RIGHT]
        self.action_space = gym.spaces.Discrete(4) # 4 discrete actions in this environment
        self.moves = {UP:(-1,0), DOWN:(1,0), LEFT:(0,-1), RIGHT:(0,1)}

        self.obs_shape = [512, 512, 3] # observation space shape, resolution & RGB

        # initialize the states
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        map_path = os.path.join(this_file_path, 'plan_ghost.txt')

        self.initial_map = self.read_map(map_path)
        self.current_map = copy.deepcopy(self.initial_map) # initialize the grid map

        self.map_shape = self.initial_map.shape
        self.height = self.map_shape[0]
        self.width = self.map_shape[1]
        self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(self.height),gym.spaces.Discrete(self.width)))

        #  initialize agent state
        self.agent_current_state = np.where(self.initial_map == AGENT)
        self.ghost_current_state = np.where(self.initial_map == GHOST)
        # target state
        self.target_state = self.get_target_state()

        self.viewer = rendering.SimpleImageViewer()
        self.emotion_values =[0,0,0,0] # joy, sadness, hope, fear values
        self.negative_reward = False
        self.postive_reward = False
        self.fear_states = []
        self.hope_states =[]
        self.real_time_reward = 0
        self.real_time_step = 0
        self.show_fear_states = False
        self.color_in_agent = GRAY
        self.ghost_posi_to_show = 0
        self.color_in_fear_posi = RED
        self.agent_current_state_in_model = 0 # only use this for show fear intensity in agent !!!

    def set_agent_current_state_in_model(self,state):# only use this for show fear intensity in agent!!!
        if type(state) == str:
            state = int(state[:-1])
        self.agent_current_state_in_model = state

    def set_ghost_posi_to_show(self,state):
        self.ghost_posi_to_show = state

    def set_color_in_fear_posi(self, rgb_color):
        self.color_in_fear_posi = rgb_color

    def set_color_in_agent(self, rgb_color):
        self.color_in_agent = rgb_color

    def target_random_reward(self):
        return -1 if np.random.uniform(0,1) < 0.3 else 20

    def set_fear_states(self, list):
        self.fear_states = list

    def set_hope_states(self, list):
        self.hope_states = list

    def set_emotion_values(self,array):
        "Set emo value [joy,sad,hope,fear]"
        self.emotion_values = array

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        return seed1

    def ghost_move_a_step(self):
        " Ghost move a random step within the chamber"
        curr_state = self.ghost_current_state

        if self.current_map[curr_state[0], curr_state[1]] == EMPTY:
            self.current_map[curr_state[0], curr_state[1]] = GHOST
        list_of_states = [48, 49, 50]
        ss = np.random.choice(list_of_states)
        ghost_next_state = self._state_to_grid_position(ss, self.width)
        next_potential_state = self.current_map[ghost_next_state[0], ghost_next_state[1]]

        if self.current_map[curr_state[0], curr_state[1]] == GHOST:
            if next_potential_state == EMPTY:
                self.current_map[ghost_next_state[0],ghost_next_state[1]] = GHOST
                # if ghost move to empty, mark current to empty
                self.current_map[curr_state[0], curr_state[1]] = EMPTY
                self.ghost_current_state = ghost_next_state
            elif next_potential_state == WALL:
                pass
            elif next_potential_state == AGENT:
                self.current_map[curr_state[0], curr_state[1]] = EMPTY
                self.ghost_current_state = ghost_next_state
        elif self.current_map[curr_state[0], curr_state[1]] == AGENT:
            if next_potential_state == EMPTY:
                self.current_map[ghost_next_state[0],ghost_next_state[1]] = GHOST
                self.ghost_current_state = ghost_next_state
            elif next_potential_state == WALL:
                pass
            elif next_potential_state == AGENT:
                self.ghost_current_state = ghost_next_state


    def _step(self, action):
        """     Run one timestep of the environment's dynamics. When end of
                episode is reached, you are responsible for calling `reset()`
                to reset this environment's state.

                Accepts an action and returns (observation, reward, done, info).

                Args:
                    action (object): an action provided by the environment

                Returns:
                    observation : agent's observation of the current environment
                    reward : amount of reward returned after previous action
                    done : whether the episode has ended
                    info : diagnostic information
        """
        action = int(action)
        reward=0
        done = False
        info = {}

        agent_next_state = (self.agent_current_state[0] + self.moves[action][0],
                            self.agent_current_state[1] + self.moves[action][1])

        next_potential_state = self.current_map[agent_next_state[0],agent_next_state[1]]
        # if agent move to empty or target, mark next as agent
        if next_potential_state == EMPTY:
            reward = self.empty_step_reward
            self.current_map[agent_next_state[0],agent_next_state[1]] = AGENT

            # if agent move to empty or target, mark current to empty
            self.current_map[self.agent_current_state[0], self.agent_current_state[1]] = EMPTY
            self.agent_current_state = agent_next_state
            info['Reach']= None
        elif next_potential_state == TARGET:
            reward = self.target_reward
            done = True
            info['Reach']='target'
            self.current_map[agent_next_state[0], agent_next_state[1]] = AGENT

            # if agent move to empty or target, mark current to empty
            self.current_map[self.agent_current_state[0], self.agent_current_state[1]] = EMPTY
            self.agent_current_state = agent_next_state
        elif next_potential_state == GHOST:
            reward = self.ghost_reward
            done = False
            info['Reach'] = 'ghost'
            self.current_map[agent_next_state[0], agent_next_state[1]] = AGENT
            self.current_map[self.agent_current_state[0], self.agent_current_state[1]] = EMPTY
            self.agent_current_state = agent_next_state
        elif next_potential_state == WALL:
            reward = self.wall_reward
            info['Reach']='wall'
            return self.current_map, reward, False, info

        return self.current_map, reward, done, info


    def simulate_step(self, state,action):
        """     return the potential next state, reward.   but not actually move the agent or change the map
        """
        action = int(action)
        grid_posi = self._state_to_grid_position(state,self.width)


        agent_next_state = (grid_posi[0] + self.moves[action][0],
                            grid_posi[1] + self.moves[action][1])
        next_potential_state = self.current_map[agent_next_state[0],agent_next_state[1]]
        reward=0

        if next_potential_state == EMPTY:
            reward = self.empty_step_reward
        elif next_potential_state == TARGET:
            reward = self.target_reward
        elif next_potential_state == WALL:
            reward = self.wall_reward
            return state, reward

        return self.grid_index_to_Statenumber(agent_next_state,self.width), reward

    def simulate_step_done(self, state,action):
        """     return the potential next state will be done or now
        """
        action = int(action)
        grid_posi = self._state_to_grid_position(state,self.width)
        agent_next_state = (grid_posi[0] + self.moves[action][0],
                            grid_posi[1] + self.moves[action][1])
        next_potential_state = self.current_map[agent_next_state[0],agent_next_state[1]]
        done = False
        if next_potential_state == TARGET:
            done = True
        return done

    def check_state_terminal(self, state):
        """     check if the state is terminal state
        """
        if state is None:
            return True
        grid_posi = self._state_to_grid_position(state,self.width)
        grid_state = self.current_map[grid_posi[0],grid_posi[1]]
        if grid_state == TARGET:
            return True
        else:
            return False

    def _state_to_grid_position(self,state,width):
        if type(state) == str:
            state = int(state[:-1])
        x = state//width
        y = np.remainder(state,width)
        return [x,y]

    def _set_agent_at(self,state):
        """
        This function set agent at empty state
        :param state: number of state ( should be empty)
        :return:
        """
        self.current_map = copy.deepcopy(self.initial_map)
        grid_position = self._state_to_grid_position(state,self.width)
        self.current_map[grid_position[0], grid_position[1]] = AGENT
        self.current_map[self.get_agent_initial_state()[0], self.get_agent_initial_state()[1]] = EMPTY
        self.agent_current_state = (grid_position[0], grid_position[1])
        self.current_map[self.ghost_ini_state()[0], self.ghost_ini_state()[1]] = EMPTY
        return self.current_map


    def _reset(self):
        self.current_map = copy.deepcopy(self.initial_map)
        self.current_map[self.ghost_ini_state()[0],self.ghost_ini_state()[1]] = EMPTY
        self.agent_current_state = self.get_agent_initial_state()
        self.current_map[self.get_agent_initial_state()[0], self.get_agent_initial_state()[1]] = AGENT
        return self.current_map

    # fill each block in RGB channel with corresponding values
    def map_to_img(self):
        obs = np.zeros(self.obs_shape)
        block_height_pixels = int(obs.shape[0]/self.height) # the height of each block in pixels
        block_width_pixels = int(obs.shape[1]/self.width)
        for i in range(self.height):
            for j in range(self.width):
                for k in range(3): # 3 RGB channels
                    color_value = COLORS[self.current_map[i,j]][k]
                    obs[i*block_height_pixels : (i+1)*block_height_pixels ,
                        j*block_width_pixels : (j+1)*block_width_pixels , k] = color_value # fill block with color
        return  (255*obs).astype(np.uint8)

    def map_to_img_show_hope_fear(self):  # before convert current map to img, change corresponding fear states
        map_to_show = copy.deepcopy(self.current_map)
        if self.fear_states:
            for state in self.fear_states:
                if type(state) == str:
                    state = int(state[:-1])
                grid_position = self._state_to_grid_position(state, self.width)
                map_to_show[grid_position[0], grid_position[1]] = FEAR

        obs = np.zeros(self.obs_shape)
        block_height_pixels = int(obs.shape[0]/self.height) # the height of each block in pixels
        block_width_pixels = int(obs.shape[1]/self.width)
        for i in range(self.height):
            for j in range(self.width):
                for k in range(3): # 3 RGB channels

                    color_value = COLORS[map_to_show[i,j]][k]
                    obs[i*block_height_pixels : (i+1)*block_height_pixels ,
                        j*block_width_pixels : (j+1)*block_width_pixels , k] = color_value # fill block with color
        return  (255*obs).astype(np.uint8)

    def map_to_img_show_fear_in_agent(self): # show fear intensity by color (from grey->red) in agent
        map_to_show = copy.deepcopy(self.current_map)
        grid_position = self._state_to_grid_position(self.agent_current_state_in_model, self.width)
        ghost_grid_posi = self._state_to_grid_position(int(self.ghost_posi_to_show), self.width)

        obs = np.zeros(self.obs_shape)
        block_height_pixels = int(obs.shape[0] / self.height)  # the height of each block in pixels
        block_width_pixels = int(obs.shape[1] / self.width)
        for i in range(self.height):
            for j in range(self.width):
                for k in range(3):  # 3 RGB channels
                    color_value = COLORS[map_to_show[i, j]][k]
                    obs[i * block_height_pixels: (i + 1) * block_height_pixels,
                    j * block_width_pixels: (j + 1) * block_width_pixels, k] = color_value  # fill block with color

        for i in range(ghost_grid_posi[0],ghost_grid_posi[0]+1):
            for j in range(ghost_grid_posi[1],ghost_grid_posi[1]+1):
                for k in range(3):  # 3 RGB channels
                    obs[i * block_height_pixels: (i + 1) * block_height_pixels,
                    j * block_width_pixels: (j + 1) * block_width_pixels, k] = COLORS[GREEN][k]

        for i in range(grid_position[0],grid_position[0]+1):
            for j in range(grid_position[1],grid_position[1]+1):
                for k in range(3):  # 3 RGB channels
                    obs[i * block_height_pixels: (i + 1) * block_height_pixels,
                    j * block_width_pixels: (j + 1) * block_width_pixels, k] = self.color_in_agent[k]

        return (255 * obs).astype(np.uint8)


    def _render(self, mode='human', sound=True, close=False):
        if mode == 'rgb_array':
            return self.map_to_img()
        elif mode is 'human' and self.show_fear_states:
            self.viewer.imshow(self.map_to_img_show_hope_fear())
        elif mode is 'human' and self.show_fear_states is False:
            self.viewer.imshow(self.map_to_img_show_fear_in_agent())

    # read map from text file
    def read_map(self, path):
        map = open(path, 'r').readlines()
        map_array = []
        for line in map:
            block_line = line.split(' ')
            line_array = []
            for block in block_line:
                try:
                    line_array.append(int(block))
                except:
                    pass
            map_array.append(line_array)
        map_array = np.array(map_array, dtype=int)
        return map_array

    def get_agent_initial_state(self):
        return np.where(self.initial_map == AGENT)

    def get_agent_current_state(self):
        return np.where(self.current_map == AGENT)

    def get_ghost_current_state(self):
        return np.where(self.current_map == GHOST)

    def ghost_ini_state(self):
        return np.where(self.initial_map == GHOST)

    def get_target_state(self):
        return np.where(self.initial_map == TARGET)

    def change_target_reward(self, reward):
        self.target_reward = reward

    def make_agent_appear_on_map(self):
        self.current_map[self.agent_current_state[0], self.agent_current_state[1]] = AGENT

    def grid_index_to_Statenumber(self,index,width):
        x = index[0]
        y = index[1]
        return (x * width) + y



