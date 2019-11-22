from math import *
import random
import copy
import gym
import gym_gridworld
import numpy as np
import pickle
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from skimage import io
import shelve

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


class Node:
    """ A node in the game tree.
    """

    def __init__(self, terminal, move=None, parent=None, state=None):
        self.terminal = terminal # if this node is terminal node or not
        self.sigma = 0.0 if terminal else 1.0
        self.move = move  # the move that make parent node to this node state , "None" for the root node
        self.state = state # state for this node
        self.Allmoves = [0, 1, 2, 3]
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.values = 0
        self.visits = 0
        self.possible_children = [] # a list of possible children could be added according to model, not node object, just[(move, state)...]
        self.unadded_possible_children = []

    def set_possible_children(self,model):
        possible_childs = []
        for m in self.Allmoves:
            trans_list = model.get_transition(self.state, m)
            if trans_list is not None:
                for pair in trans_list:
                    next_s = pair[0]
                    possible_childs.append((m,next_s))
        self.possible_children = possible_childs

    def set_unadded_possib_children(self):
        if self.childNodes == []:
            self.unadded_possible_children = self.possible_children
        else:
            for child in self.childNodes:
                move = child.move
                state = child.state
                if (move,state) in self.unadded_possible_children:
                    self.unadded_possible_children.remove((move,state))

    def UCTSelectChild(self, Q_table, model):
        """ Use the modified UCB1 formula to select a child node. If multiple max values, random choose a childnode
        """
        stochastic = False
        stocha_move = None
        child_m =[]
        for child in self.childNodes:
            move = child.move
            if move not in child_m:
                child_m.append(move)
            elif move in child_m: # if 2 child nodes have same move
                stochastic = True
                stocha_move = move

        if stochastic: # if in the child nodes, exist two child nodes with same action
            a_visit = {}
            a_Q = {}
            a_sigma = {}
            a_Q_U ={}
            stocha_child = []
            for m in child_m:
                a_visit[m] = 0 # defalut value for {action: visits}
            for child in self.childNodes: # set {action: visits} and {action: Q(s,a)}
                visit = child.visits
                move = child.move
                a_visit[move] += visit
                a_Q[move] = Q_table.get_Q(self.state, move) #Q(s,a) for 2 stocha child nodes are same, so both values are valid here
                a_sigma[move] = child.sigma
                if move == stocha_move:
                    stocha_child.append(child)

            numerator = 0
            for child in stocha_child:
                numerator += float(child.visits * child.sigma)
            denominator = 0
            for child in stocha_child:
                denominator += child.visits
            new_sigma = float(numerator/denominator)
            a_sigma[stocha_move] = new_sigma
            # choose act according to UCB + uncertanty
            value_list = []
            move_list = []
            for m in child_m:
                a_Q_U[m] = a_Q[m] + a_sigma[m] * sqrt(2* log(self.visits)/ a_visit[m])
                value_list.append(a_Q_U[m])
                move_list.append(m)
            i = self.my_argmax(np.array(value_list))
            select_move = move_list[i]

            if select_move == stocha_move: # if the chosen action is a stochastic act, select child based on prob distribution
                possi_trans = model.get_transition(self.state, select_move)
                ss = []
                possibility = []
                for pair in possi_trans:
                    next_s = pair[0]
                    prob = model.prob_of_trans(self.state, select_move, next_s)
                    ss.append(next_s)
                    possibility.append(prob)
                possibility = np.true_divide(possibility, sum(possibility))  # now all probs sum to 1
                i = np.random.choice(range(len(ss)), p=possibility)
                next_s = ss[i]
                for child in stocha_child:
                    if child.state == next_s:
                        return child
            else:
                for child in self.childNodes:
                    if select_move == child.move:
                        return  child

        else:
            value_list =[]
            child_list=[]
            for child in self.childNodes:
                Q_U_value =  Q_table.get_Q(self.state,child.move) + child.sigma*sqrt(2 * log(self.visits) / child.visits)
                child_list.append(child)
                value_list.append(Q_U_value)
            i = self.my_argmax(np.array(value_list))
            s = child_list[i]
            return s

    def my_argmax(self, array):
        """
        :param array:
        :return:  Index of the max in array
        """
        m = np.amax(array)
        indices = np.nonzero(array == m)[0]
        return random.choice(indices)

    def AddChild(self, m, s, terminal, model):
        """ Add a new child node for this move and remove this child from unadded_possible_children
            Return the added child node
        """
        n = Node(terminal, move=m, parent=self, state=s)
        n.set_possible_children(model)
        n.set_unadded_possib_children()
        self.childNodes.append(n)
        self.set_unadded_possib_children()
        return n

    def Update(self):
        """ Update this node  with a visit
        """
        self.visits += 1

    def updata_sigma(self):
        """update sigma for this node
        """
        numerator = 0
        for child in self.unadded_possible_children:
            numerator += 1
        for child in self.childNodes:
            numerator += float(child.visits * child.sigma)

        denominator = 0
        for child in self.unadded_possible_children:
            denominator += 1
        for child in self.childNodes:
            denominator += child.visits

        self.sigma = float(numerator/denominator)


class Model(object):
    """Model class that returns observation with current state and action."""

    def __init__(self):
        """Init model class"""
        self.model_transition = defaultdict(list)  # (state, action) -> list[(next_state, #of this transition)]
        self.model_reward = {} # (state, action, next_state) ->(accumuate_reward,  #of this transition)

    def set(self, state, action, next_state, reward):
        """ Store (next_state, reward) information for given (state, action).
        """
        accum_reward, num_transi = self.get_reward(state,action,next_state)
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
            return float(accume_R/num)

    def prob_of_trans(self, state, action, next_state):
        """Return the probability to next_state given (state, action),
        if no (state, action) pair in model return None
        if (state, action) pair in model exist but no next_state return 0
        """
        prob = 0
        trans_list = self.get_transition(state, action)
        if trans_list is None:
            prob = 0
        elif self.check_exist(next_state,trans_list) is False:
            prob = 0
        elif self.check_exist(next_state,trans_list) is True:
            next_state_num = 0
            tot_num = 0
            for pair in trans_list:
                if pair[0] == next_state:
                    next_state_num = pair[1]
                    tot_num += pair[1]
                else:
                    tot_num += pair[1]
            prob = float(next_state_num/tot_num)
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



def UCT(rootstate, itermax, depth, env, Q_table, gamma_value, policy, model, verbose=False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the hope_fear dict, {state: hope/fear}, each state's value are summed all traces
    :param rootstate: current state for UCT
    :param itermax:  number of trajectories ussing UCT
    :param depth: max depth for uct from root node
    :param env:  used enviroment
    :param Q_table:
    :param gamma_value: discount factor
    :param policy:  ('e-greedy', e_value) or ('softmax', t_value)
    :param model: enviroment model
    :return a dict of hope {state: hope}  if value is positive it's hope, if negative it's fear
    """

    rootnode = Node(False, state=rootstate, move= None)
    rootnode.set_possible_children(model)
    rootnode.set_unadded_possib_children()

    hope_dict ={}
    t = 0
    dep = 0
    while t < itermax and dep <= depth:
        env.reset()
        env._set_agent_at(rootstate)

        node = rootnode
        node.set_possible_children(model)
        trace = [(node.move, node.state)]
        trace_states = [node.state]



        # Select
        while (node.unadded_possible_children == [] and env.check_state_terminal(node.state) is False) and node.possible_children != []:  # node is fully expanded and non-terminal
            node = node.UCTSelectChild(Q_table, model)
            node.set_possible_children(model)

            trace.append((node.move, node.state))
            trace_states.append(node.state)
            dep = len(trace) -1


        no_possible_child = True
        # Expand
        if node.unadded_possible_children != [] and env.check_state_terminal(node.state) is False and node.possible_children != []:  # if node is non-terminal then expand
            no_possible_child = False
            expand_state =[]
            expand_move = []
            for child in node.unadded_possible_children:
                expand_move.append(child[0])
                expand_state.append(child[1])
            i = np.random.choice(len(expand_state))
            next_s = expand_state[i]
            move = expand_move[i]


            if type(next_s) == str:
                termi = False
            else:
                termi = env.check_state_terminal(next_s)
            node = node.AddChild(move, next_s, termi, model)  # add child and descend tree
            trace.append((node.move, node.state))
            trace_states.append(node.state)
            dep = len(trace) - 1

        # Backpropagate , only update visit count
        n_i = 0
        while node != None:  # backpropagate from the expanded node and work back to the root node
            if n_i ==0:
                node.Update()
            else:
                node.Update()
                node.updata_sigma()

            if trace_states.count(node.state) > 1 and n_i==0:
                node.sigma= 0.0

            node = node.parentNode
            n_i +=1

        if no_possible_child is False:
            hope_dic_this_trace = hope(trace, Q_table, gamma_value, policy, model)
            hope_dict = combine_dicts(hope_dict, hope_dic_this_trace)
            t += 1
        elif no_possible_child is True:
            t += 1



    return hope_dict  # return root node value


def combine_dicts(a, b):
    "Combine two dicts by sum values if same key"
    return {x: a.get(x, 0) + b.get(x, 0) for x in set(a).union(b)}


def hope(trace, Q_table, gamma_value, policy, model):
    """ calculate hope for the end state in one trace,
    :param trace: list of trace [(move_to_this_state, state)....]
    :param Q_table:
    :param gamma_value: discount factor
    :param policy:  ('e-greedy', e_value) or ('softmax', t_value)
    :param model: model about reward info
    :return: a dict of of [end_state:hope]
    """
    num = len(trace)
    hope_dict ={}


    prob_for_hope = 1
    dep = 1
    totalR = 0
    for i in range(num-1):
        state = trace[i][1]
        action = trace[i+1][0]
        next_s = trace[i+1][1]
        if policy[0] == 'e-greedy':
            e_value = policy[1]
            prob_for_hope = prob_for_hope * e_greedy(state,action, Q_table, e_value) * model.prob_of_trans(state,action,next_s)
        elif policy[0] == 'complete-greedy':
            prob_for_hope = prob_for_hope * complete_greedy(state, action, Q_table) * model.prob_of_trans(state,action,next_s)
        elif policy[0] == 'softmax':
            t_value = policy[1]
            prob_for_hope = prob_for_hope * softmax(state, action, Q_table, t_value) * model.prob_of_trans(state,action,next_s)
        r = model.reward_of(state,action,next_s)
        totalR += gamma_value**(dep-1) * r
        if i == num-2:
            desire = totalR + gamma_value**(dep)* np.amax(Q_table.get_array_Q(next_s)) - np.amax(Q_table.get_array_Q(trace[0][1]))
            hope_value = prob_for_hope * desire
            hope_dict[next_s] = hope_dict.get(next_s, 0) + hope_value
        dep += 1


    return hope_dict


def complete_greedy(state,action,Q_table):
    " Return the probability of chosen action at state using completely-greedy policy"
    array = Q_table.get_array_Q(state)
    max = np.amax(array)
    indices = np.nonzero(array == max)[0]
    num_of_max = len(indices)
    if action in indices:
        prob = float(1/num_of_max)
    else:
        prob = 0.0
    return prob

def e_greedy(state,action,Q_table,e_value):
    " Return the probability of chosen action at state using e-greedy policy"
    array = Q_table.get_array_Q(state)
    max = np.amax(array)
    indices = np.nonzero(array == max)[0]
    num_of_max = len(indices)
    if action in indices:
        prob = e_value*0.25 + (1-e_value)* float(1/num_of_max)
    else:
        prob = e_value*0.25
    return prob


def softmax(state,action,Q_table,t_value):
    " Return the probability of chosen action at state using softmax policy"
    prob_t = [0, 0, 0, 0]
    for a in range(4):
        prob_t[a] = np.exp(Q_table.get_Q(state, a) / t_value)
    prob_t = np.true_divide(prob_t, sum(prob_t))  # now all probs sum to 1
    return prob_t[action]


def trace_update(trace, model, Q_table, gamma_value):
    """
    After an action, and (s,a) in model, update Q value for (s,a) in trace
    :param trace: 10 most recent trace, the last added trace is at end
    :param model:
    :param Q_table:
    :param gamma_value:
    :return:
    """
    for pair in trace[::-1]:
        state = pair[0]
        action = pair[1]
        possible_trans = model.get_transition(state, action)
        sumQ = 0
        for pair in possible_trans:
            next_s = pair[0]
            prob = model.prob_of_trans(state, action, next_s)
            r = model.reward_of(state, action, next_s)
            sumQ += prob * (r + gamma_value * np.amax(Q_table.get_array_Q(next_s)))
        Q_table.set_Q(state, action, sumQ)

        lead_to_list = model.lead_to_state(state)
        for triple in lead_to_list:
            s_ = triple[0]  # (s_, a_) lead to state with prob= p
            a_ = triple[1]
            p = triple[2]
            if (s_,a_) in trace:
                pass
            else:
                rewd = model.reward_of(s_, a_, state)
                Qvalue = p * (rewd + gamma_value * np.amax(Q_table.get_array_Q(state)))
                Q_table.set_Q(s_, a_, Qvalue)

                lead_to_list2 = model.lead_to_state(s_)
                for triple2 in lead_to_list2:
                    s_2 = triple2[0]
                    a_2 = triple2[1]
                    p_2 = triple2[2]
                    if (s_2,a_2) in trace:
                        pass
                    else:
                        rewd2 = model.reward_of(s_2, a_2, s_)
                        Qvalue2 = p_2 * (rewd2 + gamma_value * np.amax(Q_table.get_array_Q(s_)))
                        Q_table.set_Q(s_2, a_2, Qvalue2)





def choose_action_greedy(sta,e_value, Q_table):
    if np.random.uniform(0,1) < e_value:
        act = random.choice([0, 1, 2, 3])
    else:
        act = argmax(Q_table.get_array_Q(sta))
    return act


def choose_action_softmax(sta, temp, Q_table):
    prob_t = [0, 0, 0, 0]
    for a in range(4):
        if Q_table.get_Q(sta,a) is None:
            prob_t[a] = np.exp(0 / temp)
        else:
            prob_t[a] = np.exp(Q_table.get_Q(sta,a) / temp)
    prob_t = np.true_divide(prob_t, sum(prob_t))  # now all probs sum to 1
    return np.random.choice([0,1,2,3], p=prob_t)


def argmax(array):
    m = np.amax(array)
    indices = np.nonzero(array == m)[0]
    return random.choice(indices)

def hope_fear_calculate(emo_dict):
    """Return hope and fear details {state: value}"""
    hope_dic = {key: value for key, value in emo_dict.items() if value > 0}
    if any(hope_dic):
        maximum = max(hope_dic.values())
        hope = {key: value for key, value in emo_dict.items() if value == maximum}
    else:
        hope = {}

    fear_dic = {key: value for key, value in emo_dict.items() if value < 0}
    if any(fear_dic):
        minimum = min(fear_dic.values())
        fear = {key: value for key, value in emo_dict.items() if value == minimum}
    else:
        fear = {}
    return hope,fear


def joy_sad_calculate(previous_state, previous_act, current_state, rewar, Q_table, gamma_value):
    """calculate joy/sad based on actual reward"""
    if Q_table.get_Q(previous_state,previous_act) is None:
        joy_sad = rewar + gamma_value * np.amax(Q_table.get_array_Q(current_state)) - 0
    else:
        joy_sad = rewar + gamma_value*np.amax(Q_table.get_array_Q(current_state)) - Q_table.get_Q(previous_state,previous_act)
    if joy_sad >=0:
        joy = joy_sad
        sad = 0
    else:
        sad = joy_sad
        joy = 0
    return joy,sad
######################################
# Record the one version of agent(e-greedy decay) moving with relevant info without actual fear calculation
start_time = time.time()
env = gym.make('Gridworld-v0')

nS = np.prod([env.height,env.width]) # number of states
nA = len(env.actions) # number of actions

Q = Qtable()
model = Model()

env._reset()
state = env.grid_index_to_Statenumber(env.agent_current_state, env.width)[0]

t = 0
max_steps = 500
uct_iterations = 10
uct_depth = 2

joy_value =0
sad_value = 0
hope_value = 0
fear_value =0

explore_steps = 300

e_list = np.linspace(0.5, 0.1, num=explore_steps, endpoint=True)


reward_list_for_plot = []
fear_list_for_plot = []
total_fear ={}
total_hope ={}
session_trace = []
with shelve.open('egreedy_decay', 'c') as shelf:
    for i in range(max_steps):
        shelf[str(i)] = {'Q': Q, 'model before act': model, 'model after act': model, 'agent state': state,
                         'agent act': 0, 'act reward': 0, 'reach target': False, 'agent state after act': 0}

    while t < max_steps:
        env.real_time_step = t
        if t < explore_steps:
            policy = ('e-greedy', e_list[t])
        else:
            policy = ('e-greedy', 0.1)


        temp = shelf[str(t)]
        temp['agent state'] = state
        temp['Q'] = Q
        temp['model before act'] = model
        shelf[str(t)] = temp


        hope_fear = UCT(state, uct_iterations, uct_depth, env, Q, 0.9, policy, model, verbose=False)
        hope_list, fear_list = hope_fear_calculate(hope_fear)


        if any(fear_list):
            fear_value = list(fear_list.values())[0]  # fear_list contains all max fear, so if exist any value is ok to use
            fear_states = list(fear_list.keys())
            env.set_fear_states(fear_states)
        else:
            fear_value = 0
            env.set_fear_states([])

        if any(hope_list):
            hope_value = list(hope_list.values())[0]
            hope_states = list(hope_list.keys())
        else:
            hope_value = 0


        env.set_emotion_values([joy_value, sad_value, hope_value, fear_value])
        fear_list_for_plot.append(fear_value)
        env.make_agent_appear_on_map()


        env.ghost_move_a_step()


        if policy[0] == 'e-greedy':
            action = choose_action_greedy(state, policy[1], Q)
        elif policy[0] == 'softmax':
            action = choose_action_softmax(state, policy[1], Q)

        temp = shelf[str(t)]
        temp['agent act'] = action
        shelf[str(t)] = temp


        if len(session_trace) > 10:
            del session_trace[0]
            session_trace.append((state,action))
        else:
            session_trace.append((state,action))


        total_fear = combine_dicts(total_fear, fear_list)
        total_hope = combine_dicts(total_hope, hope_list)

        env.postive_reward = False
        env.negative_reward = False

        obs, reward, done, info = env.step(action)

        temp = shelf[str(t)]
        temp['act reward'] = reward
        temp['reach target'] = done
        shelf[str(t)] = temp


        reward_list_for_plot.append(reward)
        env.real_time_reward = reward
        if reward < 0 :
            env.negative_reward = True
            env.postive_reward = False
        elif reward >0:
            env.negative_reward = False
            env.postive_reward = True


        state2 = env.grid_index_to_Statenumber(env.agent_current_state, env.width)  #actual position on map

        if info['Reach'] == 'ghost':
            ns = str(state2) + 'D'
            model.set(state, action, ns, reward)
            state2 = ns
        else:
            model.set(state,action,state2,reward)

        temp = shelf[str(t)]
        temp['agent state after act'] = state2
        shelf[str(t)] = temp

        temp = shelf[str(t)]
        temp['model after act'] = model
        shelf[str(t)] = temp
        trace_update(session_trace, model, Q, 0.9)

        state = state2

        if done:
            env.reset()
            state = env.grid_index_to_Statenumber(env.agent_current_state, env.width)[0]
            session_trace = []


        t +=1



print("--- %s seconds ---" % round(time.time() - start_time, 2))



