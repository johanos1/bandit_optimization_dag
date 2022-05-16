import argparse
from matplotlib import pyplot as plt
import numpy as np
import multiprocessing as mp
import pandas
import random
from games import SocioEconomicGame

random.seed(10) # set random seed


class GameInstance:
   def __init__(self, T=10000, setting = 'decentralized'):
      self.T = T
      if setting == 'decentralized':
         self.game = SocioEconomicGame(self.T)   
      else:
         raise Exception("The setting is not recognized")         
   
   def simulate(self, K=10):
      print("--------------Simulating game-----------------")
      self.play_game(K)
      N = mp.cpu_count()
      with mp.Pool(processes=N) as p:
         results = p.map(self.play_game, list(range(K)))
      
      reward_trajectories = np.vstack(results)
      return reward_trajectories
      
   
   def play_game(self, K):
      
      # allocate space to keep track of everything
      observed_rewards = np.empty(self.T)
      self.game.reset_game()
      for t in range(self.T):
         if t % 1e5 == 0 and t > 0:
            print(f"Iteration {t}")
         observed_rewards[t] = self.game.step()
         self.game.update_policies(observed_rewards[t])

      actions = self.game.get_most_likely_actions()
      print(f"Most likely actions of instance {K} is: {actions}")
      return observed_rewards

   def evaluate_upper_bound(self):
      print("--------------Obtaining upper bound on regret-------------- \n")
      regret_ub = self.game.regret_upper_bound()
      return regret_ub
   
   def evaluate_regret(self, reward_trajectories):
      print("\n--------------Searching for best reward in hindsight--------------")
      optimal_reward, best_actions = np.asarray(self.game.get_best_rewards_in_hindsight())
      print(f"the best actions in hindsight are: {best_actions} \n")
      
      # Evaluate the pseudo regret
      cumulative_rewards = np.cumsum(reward_trajectories, axis=1)
      mean_cumulative_rewards = np.mean(cumulative_rewards, axis=0)
      cumulative_rewards_hindsight = np.cumsum(optimal_reward)
      pseudo_regret = (cumulative_rewards_hindsight - mean_cumulative_rewards)/range(1,self.T+1)
      
      std_pseudo_regret = np.std(cumulative_rewards_hindsight - np.cumsum(reward_trajectories, axis=1), axis=0)/range(1,self.T+1) 
      
      return pseudo_regret, std_pseudo_regret, cumulative_rewards, cumulative_rewards_hindsight
          
      

def save_data(data):
   df = pandas.DataFrame(data).T
   indices = np.unique(np.round(np.logspace(0,6,1000)))
   df = df[df[df.columns[0]].isin(indices)]
   df.to_csv('output.csv', sep=',')
   

def main(args):
   np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
   
   T = args.T
   K = args.K
   T = 10000
   K=10
   setting = 'decentralized'
   game_instance = GameInstance(T, setting)
   regret_ub = game_instance.evaluate_upper_bound()
   reward_trajectories = game_instance.simulate(K)
   pseudo_regret, std_pseudo_regret, cumulative_rewards, cumulative_rewards_hindsight = game_instance.evaluate_regret(reward_trajectories)
   

   # --------------   Plot the results   -------------- 
   upper_lim = (pseudo_regret + 2*std_pseudo_regret)
   lower_lim = (pseudo_regret - 2*std_pseudo_regret)
   
   # plot rewards
   fig, (ax1, ax2) =plt.subplots(1,2)
   for a in cumulative_rewards:
      ax1.plot(range(1,T+1), a/range(1,1+T))
   
   ax1.plot(range(1,T+1), cumulative_rewards_hindsight/range(1,1+T), '--', label='Cumulative rewards in hindsight')
   ax1.set_xscale("log")
   ax1.set(xlabel='Rounds, T', ylabel='Reward')
   ax1.legend(loc="lower right")

   # Plot the pseudo regret
   ax2.fill_between(range(1,T+1),lower_lim,upper_lim, color='C0', alpha=0.3,)
   ax2.plot(range(1,T+1), pseudo_regret,  label='Pseudo regret')
   ax2.plot(range(1,T+1), regret_ub,  label='Upper bound')
   ax2.set_xscale("log")
   ax2.set(xlabel='Rounds, T', ylabel='Regret')
   ax2.legend(loc="upper right")
   ax2.set_xlim([1, T])
   ax2.set_ylim([0, 1])

   plt.suptitle(setting)
   plt.show()

   results = (range(1,T+1), pseudo_regret, lower_lim, upper_lim, regret_ub)
   save_data(results)
   
if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='Personal information')
   parser.add_argument('--T', dest='T', type=int, help='Number of rounds')
   parser.add_argument('--K', dest='K', type=int, help='Number of realizations')
   args = parser.parse_args()
   main(args)



