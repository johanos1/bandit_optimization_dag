import numpy as np
import scipy.optimize as opt
from typing import List, Optional, Set, Tuple, Union

# Ref: https://stackoverflow.com/a/56641168/
def ensure_list(s: Optional[Union[str, List[str], Tuple[str], Set[str]]]) -> List[str]:
    return s if isinstance(s, list) else list(s) if isinstance(s, (tuple, set)) else [] if s is None else [s]


class TsallisInf:
    def __init__(self, num_actions):
        self.num_actions = num_actions # number of actions available to the player
        self.alpha = 0.5 # alpha value in the Tsallis-Inf algorithm
        self.cumulative_losses = np.zeros(self.num_actions) # Keep track of the losses observed for each action
        self.last_played_action = -1 # keep track of latest action
        self.weights = np.full((self.num_actions), 1/self.num_actions) # current strategy
    
    
    def sample_action(self):
        """Sample an action based on the strategy."""
        self.last_played_action = np.random.choice(a=np.arange(self.num_actions), p=self.weights)
        return self.last_played_action
    
    # Inspired by https://smpybandits.github.io/_modules/Policies/TsallisInf.html
    def update_policy(self, reward, time):
        """Update the strategy based on the observed reward using the Tsallis-Inf algorithm"""
        # for a reward in [0,1], loss = 1 - reward
        biased_loss = 1.0 - reward
        # unbiased estimate, from the weights of the previous step
        unbiased_loss = biased_loss / self.weights[self.last_played_action]
        self.cumulative_losses[self.last_played_action] += unbiased_loss
        eta_t = 1.0 / np.sqrt(max(1,time))
        
        # solve f(x)=1 to get an approximation of the (unique) Lagrange multiplier x
        def objective_function(x):
            return (np.sum( (eta_t * (self.cumulative_losses - x + np.finfo(float).eps)) ** -2) - 1)**2 
        result_of_minimization = opt.minimize_scalar(objective_function)
        x = result_of_minimization.x
        #  use x to compute the new weights
        new_weights =  ( eta_t * (self.cumulative_losses - x) ) ** -2
        
        # Bad case, where the sum is so small that it's only rounding errors
        # or where all values where bad and forced to 0, start with new_weights=[1/K...]
        if not np.all(np.isfinite(new_weights)):
            new_weights[:] = 1.0
        # 3. Renormalize weights at each step
        new_weights /= np.sum(new_weights)
        # 4. store weights
        self.weights =  new_weights
        
    def reset(self):
        """Reset the strategy to the uniform strategy"""
        self.weights = np.full((self.num_actions), 1/self.num_actions);       
        self.cumulative_losses = np.zeros(self.num_actions) # Keep track of the losses observed for each action
        self.last_played_action = -1 # keep track of latest action
             
    def get_policy(self):
        """Read out the policy"""
        return self.weights
    

class Player:  
    def __init__(self, num_actions, leaders):
        self.num_actions = num_actions # actions available to the player
        self.learner_indx = -1 # this is the index of the learning algorithm to be updated
        self.indx = len(leaders) # the index of the player in the hierarchy
        self.num_leader_actions = [leader.num_actions for leader in leaders] # find out the number of actions of each player going earlier
        self.num_joint_leader_actions = np.prod(self.num_leader_actions) # the size of the joint action space of players going earlier

        # Create as many learners as there are joint actions of the leading players
        self.learners = [] # list to keep track of the learners
        if self.indx > 0: # there are other players before
            self.counters = np.full(self.num_joint_leader_actions, 0) # keep track of the number of times the learners have been called
            for i in range(self.num_joint_leader_actions): # for each joint leader actions, create a learner
                self.learners.append(TsallisInf(self.num_actions))        
        else: # if there are no players going earlier
            self.counters = np.full(1, 0) # a single counter is enough
            self.learners.append(TsallisInf(self.num_actions))   # create a single learner
    
    def reset(self):
        """Reset all the learners of the player"""
        if self.indx > 0: 
            self.counters = np.full(self.num_joint_leader_actions, 0) 
        else:
            self.counters = np.full(1, 0)
            
        for learner in self.learners:
            learner.reset()
    
    def observe_action(self, leader_actions):
        """Observe the actions of the players before and find out the corresponding learner index"""
        # If the current player is the leader, there are no actions to observe
        if self.indx == 0:
            self.learner_indx = 0
            self.counters += 1
        else:
            # The leaders' actions are viewed as an N-dimensional matrix. The corresponding bandit index is the linear index of the observed actions
            # ravel_multi_index takes as inputs: one integer for each dimension and the dimensions
            leader_actions = ensure_list(leader_actions)
            self.learner_indx = np.asscalar(np.ravel_multi_index(leader_actions, dims= tuple(self.num_leader_actions), order='C')) # find the learner index of the given actions
            self.counters[self.learner_indx] += 1 # increase the counter for the learner
    
    def pick_action(self):   
        """Sample an action according to the learner strategy"""
        return self.learners[self.learner_indx].sample_action()
    
    def update_policy(self, reward):
        """Update policy according to reward"""
        if self.learner_indx >= 0: # if an action has been observed
            time = self.counters[self.learner_indx] # find out how many times the current learner has been called
            self.learners[self.learner_indx].update_policy(reward, time) # update the current learner based on the reward
        else:
            print(f"The learner has not been set for player {self.indx}")
        
    def get_policy(self):
        """Read out policy of the current learner"""
        return self.learners[self.learner_indx].get_policy()