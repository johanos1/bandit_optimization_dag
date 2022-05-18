# Decentralized Online Bandit Optimization on Directed Graphs with Regret Bounds
This repository contains the source code to generate the numerical results in the paper _Decentralized Online Bandit Optimization on Directed
Graphs with Regret Bounds_. The simulation scenario is inspired by [_The AI economist_](https://arxiv.org/abs/2108.02755). In particular, there are $M+1$ players where one player acts as a socio-economic planner and the remaining players act as workers. The game is played over T rounds and proceeds as follows over each round:
1. The socio-economic planner decides for a taxation policy
2. The workers observe the taxation policy and pick actions consecutively
3. Each worker action is mapped to an income and a labor cost
4. The net income of each worker is obtained by subtracting the tax collectedby the socio-economic planner
5. The worker utility is decided from the net income and the labor cost
6. The bandit reward is a weighted average of all worker utilities and the collected tax (all normalized to [0,1])
7. All the players observe the bandit reward and update their respective policies.


To generate Fig.3 in the paper, run:
```
python3 main.py --T 1000000 --K 100
``` 
This command will store the result in output.csv and generate the figure below


<img src="https://user-images.githubusercontent.com/40794255/168978565-76062882-c359-4b6c-b075-5cfc483eb289.png" width="100" height="100">
