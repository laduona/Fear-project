import numpy as np
import pickle
import shelve
from collections import defaultdict


class Qtable(object):
    def __init__(self):
        """Init model class"""
        self.Q_table = {} # (state, action) -> (value)

    def set_Q(self, state, action, value):
        if self.get_Q(state,action) is None:
            for a in range(4):
                self.Q_table[(state, a)] = 0
            self.Q_table[(state, action)] = value
        else:
            self.Q_table[(state, action)] = value

    def get_Q(self, state, action):
        " if no such (s,a) return None"
        return self.Q_table.get((state,action))

    def get_array_Q(self, state):
        "Return a array of Q[s,:] by sequence"
        if self.get_Q(state, 0) is None: # check if there are record for state in Q_table
            for a in range(4):
                self.Q_table[(state, a)] = 0
        arra = [0, 0, 0, 0]
        a_values ={}
        for s,a in self.Q_table.keys():
            if s == state:
                a_values[a] = self.Q_table[(s,a)]
        for m in range(4):
            arra[m] = a_values[m]
        return arra

class Model(object):
    """Model class that returns observation with current state and action."""

    def __init__(self):
        """Init model class"""
        self.model_transition = defaultdict(list)  # (state, action) -> list[(next_state, #of this transition)]
        self.model_reward = {}  # (state, action, next_state) ->(accumuate_reward,  #of this transition)

    def set(self, state, action, next_state, reward):
        """ Store (next_state, reward) information for given (state, action).
        """
        accum_reward, num_transi = self.get_reward(state, action, next_state)
        if num_transi == 0:
            self.model_reward[(state, action, next_state)] = (reward, 1)
        else:
            num_transi += 1
            accum_reward += reward
            self.model_reward[(state, action, next_state)] = (accum_reward, num_transi)

        possible_trans = self.get_transition(state, action)
        if (possible_trans is None) or self.check_exist(next_state, possible_trans) is False:
            self.model_transition[(state, action)].append((next_state, 1))
        elif (possible_trans is not None) and self.check_exist(next_state, possible_trans) is True:
            self.update_trans_list(state, action, next_state)

    def update_trans_list(self, state_update, action_update, next_state):
        "Update the # in (next_state, #of this transition) by 1"
        possible_trans = self.get_transition(state_update, action_update)
        num = 0
        for pair in possible_trans:
            if pair[0] == next_state:
                num = pair[1]
                self.get_transition(state_update, action_update).remove((next_state, num))
            else:
                pass
        num += 1
        self.get_transition(state_update, action_update).append((next_state, num))

    def check_exist(self, state, list):
        "Check if state is in the list of possible transisions [(next_state, #of this transition)] in model_transition for (state, action)"
        exist = False
        for pair in list:
            next_stae = pair[0]
            if next_stae == state:
                exist = True
                break
            else:
                pass
        return exist

    def get_transition(self, state, action):
        """ Returns list of possible transisions [(next_state, #of this transition)] stored for (state, action).
            If no entry at all found for (state, action), it returns None.
        """
        return self.model_transition.get((state, action))

    def get_reward(self, state, action, next_state):
        """ Returns (accumuate_reward, #of this transition) stored for (state, action, next_state).
            If no entry found for (state, action, nexxt_state), it returns (0,0).
        """
        return self.model_reward.get((state, action, next_state), (0, 0))

    def reward_of(self, state, action, next_state):
        " Return the average reward for (state, action, next_state)"
        accume_R, num = self.get_reward(state, action, next_state)
        if num == 0:
            return 0
        else:
            return float(accume_R / num)

    def prob_of_trans(self, state, action, next_state):
        """Return the probability to next_state given (state, action),
        if no (state, action) pair in model return None
        if (state, action) pair in model exist but no next_state return 0
        """
        prob = 0
        trans_list = self.get_transition(state, action)
        if trans_list is None:
            prob = 0
        elif self.check_exist(next_state, trans_list) is False:
            prob = 0
        elif self.check_exist(next_state, trans_list) is True:
            next_state_num = 0
            tot_num = 0
            for pair in trans_list:
                if pair[0] == next_state:
                    next_state_num = pair[1]
                    tot_num += pair[1]
                else:
                    tot_num += pair[1]
            prob = float(next_state_num / tot_num)
        return prob

    def lead_to_state(self, lead_to_state):
        "Return a list of (state,action,prob) that lead to state"
        lead_to_list = []
        for key in self.model_transition:
            list = self.model_transition[key]
            state = key[0]
            action = key[1]
            for pair in list:
                next_state = pair[0]
                if next_state == lead_to_state:
                    prob = self.prob_of_trans(state, action, next_state)
                    lead_to_list.append((state, action, prob))
        return lead_to_list

    def reset(self):
        "Clear the model. Delete all stored (state, action) pairs."
        self.model_transition.clear()
        self.model_reward.clear()

############################
# store when and where the agent hits the ghost for visualization in ghost_states.pkl
ghost_hit_state = {}
with shelve.open('egreedy_decay', 'r') as shelf:
    for t in range(500):
        state = shelf[str(t)]['agent state']
        state_after = shelf[str(t)]['agent state after act']
        if type(state_after) == str:
            print('Step: ',t)
            print('Ghost at:', state_after)
            ghost_hit_state[t] = int(state_after[:2])

ghost_possible_states = [48, 49, 50]
ghost_states = np.zeros(500)

for i, state in enumerate(ghost_states):
    ghost_states[i] = np.random.choice(ghost_possible_states)

for key in ghost_hit_state:
    ghost_states[key] = ghost_hit_state[key]

with open('ghost_states-test.pkl', 'wb') as f:
    pickle.dump(ghost_states, f)