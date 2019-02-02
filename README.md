# Bandits

For the moment two algorithms are implemented :
  - Epsilon Greedy
  - UCB (Upper Confidence Bound)
 
THe class KarmedBandits creates an environment of k armed.
The class monitor plots the regret or the cumulative reward for a list of policies. A policy needs to be a class with the same methods as the ones implemented for EpsGreedy and some attributes (essentialy the list of actions and rewards).

