import numpy as np
from  .players import Player
from itertools import product as iterproduct
import abc

class Game:
    """A class used to represent a leader-follower game"""
    def __init__(self, T, num_players, num_player_actions):
        self.T = T # number of rounds
        self.num_players = num_players # number of players
        self.num_player_actions = num_player_actions # number of actions
        self.players = [] # list to store player objects
        self.create_players() # populate the list with players taking into account the leader-follower hierarchy
    
    def create_players(self):
        """Create players based on leader-follower structure"""
        if self.num_players == 1:
            self.players.append(Player(self.num_player_actions, self.players))
        else:
            for i in range(self.num_players):
                self.players.append(Player(self.num_player_actions[i], self.players)) # create a player that is aware of the actions players going before may take
    
    def get_player_actions(self):
        """Sample an action from all the players"""
        actions = [] # list to store the actions
        for i in range(self.num_players):  # go through all the players
            self.players[i].observe_action(actions) # let the player observe the actions of the earlier players
            actions.append(self.players[i].pick_action()) # based on the observed actions, let the player pick an action
        return actions # return the joint action
    
    def reset_game(self):
        """Reset the game"""
        for player in self.players: # reset each player
            player.reset()
            
    @abc.abstractmethod
    def get_reward(self):
        """Abstract method that is game specific"""
        return
    
    @abc.abstractmethod
    def regret_upper_bound(self):
        """Abstract method to evaluate the upper bound on the pseudo regret"""
        return


class SocioEconomicGame(Game):
    """Socioeconomic game implementation"""
    def __init__(self, T):
        # prepare the planner
        self.income_brackets = [0,14,1e10]
        self.marginal_taxes = [0.1, 0.3, 0.5] # actions of each leader
        self.num_income_brackets = len(self.income_brackets)-1 # how many brackets do we want
        self.actions_per_bracket = len(self.marginal_taxes) # how many marginal taxes to choose between
        self.planner_actions = self.actions_per_bracket  ** self.num_income_brackets 
        self.action_to_taxrate = list(iterproduct(self.marginal_taxes, repeat = self.num_income_brackets)) # get list of all different tax rates
        
        # create players
        num_workers = 3
        num_players = num_workers + 1
        self.num_worker_actions = 3
        
        M= num_workers
        self.w = 1/M
        self.w_l = M
        
        # create list with size of action spaces for all players
        num_player_actions = [self.planner_actions]
        for i in range(num_workers):
            num_player_actions.append(self.num_worker_actions)
        
        # Initiate the game and create players
        super().__init__(T, num_players, num_player_actions)
        self.total_collected_taxes = 0 # total collected taxes
        
        # prepare workers
        # create mapping from action to (marginal gross income, marginal labour)
        self.default_incomes = [5, 10, 15]
        self.gamma = 0.3 # nonlinearity in utility function
        
        # initiate the workers
        for i in range(1, len(self.players)):
            self.players[i].cumulative_income = 0 # each worker starts with no cumulative income
            self.players[i].cumulative_labour = 0 # each worker starts with no cumulative labor
            skill = i # the skill level is given as the worker index
            self.players[i].action_map = []  # initiate mapping from action to (marginal income, marginal labour)
            for j in range(self.num_worker_actions): # Go through each worker action
                marginal_income = self.default_incomes[j]  # marginal income depends on the skill
                marginal_labour = j / skill # higher income means higher labour
                self.players[i].action_map.append(tuple([marginal_income, marginal_labour]))  # save mapping for each action
        
        # Make sure all variables match
        assert  self.planner_actions == len(self.action_to_taxrate), "actions and mapping to tax rates doesn't match"
        assert  self.num_worker_actions == len(self.default_incomes), "actions and default incomes do not match"
        assert M*self.w + self.w_l == M+1, "weights in reward do not add up"
        
        
    def reset_game(self):
        """Reset the game"""
        super().reset_game() # reset each player
        self.total_collected_taxes = 0 # make collected tax zero
        # make cumulative income and labour of each player zero
        for player in self.players:
            player.cumulative_income = 0
            player.cumulative_labour = 0
    
    
    def get_reward(self, actions):
        """Evaluate the bandit reward for the players"""
        
        def collect_tax(income, tax_rates):
            """Collect tax from a worker"""
            # Tax the worker's gross income
            collected_tax = 0
            for i in range(self.num_income_brackets): # go through each income bracket
                tax_rate = tax_rates[i] # pick the tax rate for the bracket
                if income > self.income_brackets[i+1]: # is if the income is larger than the upper limit of the bracket
                    collected_tax += tax_rate*(self.income_brackets[i+1] - self.income_brackets[i] ) # compute tax
                elif income > self.income_brackets[i]: # if it falls inside the tax bracket
                    collected_tax += tax_rate*(income-self.income_brackets[i]) # only tax the overlapping part
                else:
                    pass
            return collected_tax 
        
        # Compute worker utilities
        def worker_utility(cumulative_income, cumulative_labour):
            """Compute utility of a worker"""
            return (np.float_power(cumulative_income, 1-self.gamma) - 1) / (1-self.gamma) - cumulative_labour
        
        def find_min_max_utility(player):
            """Given the cumulative income and labour from previous rounds, find out the max/min achievable utility"""
            max_utility = -np.inf
            min_utility = np.inf
            for j in range(self.num_worker_actions): # go through each worker action
                marginal_income = player.action_map[j][0] # get the marginal income for the action
                marginal_labour = player.action_map[j][1] # get the marginal labour for the action
                net_income = marginal_income - collect_tax(marginal_income, tax_rates) # compute the net income
                cum_inc = player.cumulative_income + net_income # get the cumulative income for the action
                cum_lab = player.cumulative_labour + marginal_labour # get the cumulative labour for the action
                u = worker_utility(cum_inc, cum_lab) # compute worker utility
                if u > max_utility: max_utility = u # if larger than current max, save it
                if u < min_utility: min_utility = u # if smaller than current min, save it
            return min_utility, max_utility
        
        # get planner action and corresponding tax rate
        tax_rates = self.action_to_taxrate[actions[0]]
        # tax all players
        collected_taxes = []
        utilities = []
        
        # go through each player to compute the normalized utility
        for i in range(1, len(self.players)):
            player = self.players[i]
            # get the marginal income and labour for the played action
            marginal_income = player.action_map[actions[i]][0]
            marginal_labour = player.action_map[actions[i]][1] 
            # evaluate the tax for the action
            collected_taxes.append(collect_tax(marginal_income, tax_rates))
            # compute net income
            net_income = marginal_income - collected_taxes[-1]
            # evaluate the max and min utilities possible for the player in the current round
            min_utility, max_utility = find_min_max_utility(player)
            # get cumulative income and labour
            player.cumulative_income += net_income
            player.cumulative_labour = np.max(marginal_labour + player.cumulative_labour, 0)
            # evaluate the worker utility
            player_utility = worker_utility(player.cumulative_income, player.cumulative_labour)
            # normalize the utility to [0,1] and save it
            if max_utility == min_utility:
                utilities.append(1.0)
            else:
                utilities.append((player_utility - min_utility) / (max_utility - min_utility))
           
            
            
        # normalize all rewards to make it between [0,1]
        max_tax = 0
        min_tax = 0
        min_tax_rates = np.full(self.num_income_brackets, np.min(self.marginal_taxes))
        max_tax_rates = np.full(self.num_income_brackets, np.max(self.marginal_taxes))
        
        for j in range(1,len(self.players)):     
            marginal_income = self.players[j].action_map[actions[j]][0]
            max_tax += np.sum(collect_tax(marginal_income, max_tax_rates))  
            min_tax += np.sum(collect_tax(marginal_income, min_tax_rates)) 
        max_tax += self.total_collected_taxes 
        min_tax += self.total_collected_taxes
        # actual collected taxes and normalize it
        self.total_collected_taxes += np.sum(collected_taxes)
        if max_tax == min_tax:
            normalized_collected_tax = 1
        else:
            normalized_collected_tax = (self.total_collected_taxes - min_tax) / (max_tax - min_tax)
        
        
        # compute the reward
        r = (1/(len(self.players))) * (self.w*np.sum(utilities) + self.w_l*normalized_collected_tax)
        
        return r
    
    def step(self):   
        """Step through one round of the game"""
        # sample actions
        actions = super().get_player_actions()
        observed_reward = self.get_reward(actions)
        return observed_reward
    
    def update_policies(self, r):
        """Update players policies based on observed reward"""
        for i in range(self.num_players):  
            self.players[i].update_policy(r)
        
    def regret_upper_bound(self):
        """Upper bound on the pseudo regret"""
        T_ub = range(1,self.T+1)
        num_actions = [self.planner_actions]
        for player in self.players:
            num_actions.append(self.num_worker_actions)
        
        # the tax part is using all actions:
        tax_term = 4*np.sqrt(T_ub)*np.sum(np.sqrt(np.cumprod(num_actions))) + np.sum( np.cumprod(num_actions[:-1]) ) + 1
        
        # The worker parts only contain a Stackelberg game 
        worker_part = 4*np.sqrt(T_ub)*np.sum(np.sqrt(np.cumprod(num_actions[0:2]))) + num_actions[0] + 1
        
        # the upper bound is given by 
        regret_ub = (tax_term + (len(self.players)-1)*worker_part) / len(self.players)
        
        upper_bound = regret_ub/T_ub
        return upper_bound
    
    def get_most_likely_actions(self):
        actions = []
        for i, player in enumerate(self.players):
            if i > 0:
                player.observe_action(actions)
            policy = player.get_policy()
            max_indx = np.argmax(policy) # extract the most likely action of current player
            actions.append(max_indx) 
        return actions
            
    def get_best_rewards_in_hindsight(self):
        """Obtain the best determinisitc joint action in hindsight"""
        # loop through all different actions available and collect the best result
        num_workers = self.num_players-1
        worker_actions = list(range(self.num_worker_actions))
        worker_joint_actions = list(iterproduct(worker_actions, repeat = num_workers)) # get list of all different worker actions
        
        best_actions = []
        best_cumulative_reward = 0
        for a in range(self.planner_actions):
            for b in worker_joint_actions:
                self.reset_game()
                actions = []
                actions.append(a) 
                actions += list(b)
                reward = 0
                for t in range(1,np.minimum(10000, self.T)): # this is used to speed up the search
                    reward += self.get_reward(actions)
                if reward > best_cumulative_reward: 
                    best_cumulative_reward = reward
                    best_actions = actions
        
        # obtain all the rewards for the best actions
        reward=[]
        self.reset_game()
        for t in range(self.T):
            reward.append(self.get_reward(best_actions))       
        return reward, best_actions
