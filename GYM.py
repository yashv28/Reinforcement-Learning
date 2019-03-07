import numpy as np
import gym
import itertools
import matplotlib
import sys
import sklearn.pipeline
import sklearn.preprocessing
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

#####################################################

mc = True # True - Mountain Car, False - Acrobot

iters = 100
epsilon = 0.0
discount_factor = 0.99 
epsilon = 0.1 
epsilon_decay = 1.0

#####################################################

if mc:
	env = gym.envs.make("MountainCar-v0")
else:
	env = gym.envs.make("Acrobot-v1")

itersGraph = namedtuple("Graphs",["iter_lengths", "iter_rewards"])

# Take samples as inputs for RBF kernels approximation of states
samples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(samples)

approximate = sklearn.pipeline.FeatureUnion([
		("rbf1", RBFSampler(gamma=5.0, n_components=100)),
		("rbf2", RBFSampler(gamma=2.0, n_components=100)),
		("rbf3", RBFSampler(gamma=1.0, n_components=100)),
		("rbf4", RBFSampler(gamma=0.5, n_components=100))
		])
approximate.fit(scaler.transform(samples))

# Featurizes states over different RBF kernels
def predict(s, a=None):
	scaled = scaler.transform([s])
	features = approximate.transform(scaled)[0]
	if not a:
		return np.array([m.predict([features])[0] for m in models])
	else:
		return models[a].predict([features])[0]

def epsilonGreedyPolicy( epsilon, nA):
	def policy_fn(observation):
		A = np.ones(nA, dtype=float) * epsilon / nA
		q_values = predict(observation)
		best_action = np.argmax(q_values)
		A[best_action] += (1.0 - epsilon)
		return A
	return policy_fn


models = []
for _ in range(env.action_space.n):
	model = SGDRegressor(learning_rate="constant")
	model.partial_fit([approximate.transform(scaler.transform([env.reset()]))[0]], [0])
	models.append(model)

graph = itersGraph(iter_lengths=np.zeros(iters),iter_rewards=np.zeros(iters))    
	
# Q-learning
for i in range(iters):

	policy = epsilonGreedyPolicy(epsilon * epsilon_decay**i, env.action_space.n)
	
	last_reward = graph.iter_rewards[i - 1]
	sys.stdout.flush()
	
	state = env.reset()
	
	next_action = None
	
	for t in itertools.count():
		if next_action is None:
			action_probs = policy(state)
			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
		else:
			action = next_action
		
		next_state, reward, done, _ = env.step(action)

		graph.iter_rewards[i] += reward
		graph.iter_lengths[i] = t

		a = None # Set a for some action, else None for all actions.
		if not a:
			Qvalnext = np.array([m.predict([approximate.transform(scaler.transform([next_state]))[0]])[0] for m in models])
		else:
			Qvalnext = models[a].predict([approximate.transform(scaler.transform([next_state]))[0]])[0]
		
		td_target = reward + discount_factor * np.max(Qvalnext)
		
		models[action].partial_fit([approximate.transform(scaler.transform([state]))[0]], [td_target])
			
		if done:
			break
			
		state = next_state

	print "Episode {0}/{1} (Last Reward = {2})".format(i + 1, iters, last_reward)

if mc:
	st = "MountainCar-v0"
else:
	st = "Acrobot-v1"

fig1 = plt.figure(figsize=(10,5))
plt.plot(graph.iter_lengths)
plt.xlabel("#Iterations")
plt.ylabel("Iteration Length")
plt.savefig('Results/'+st+"_iteration_length.png")
plt.show(fig1)

fig2 = plt.figure(figsize=(10,5))
rewards_smoothed = pd.Series(graph.iter_rewards).rolling(25, min_periods=25).mean()
plt.plot(rewards_smoothed)
plt.xlabel("#Iterations")
plt.ylabel("Reward")
plt.savefig('Results/'+st+"_rewards.png")
plt.show(fig2)
