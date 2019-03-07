# University of Pennsylvaina
# ESE650 Fall 2018 
# Heejin Chloe Jeong

import numpy as np

def get_action_egreedy(values ,epsilon):
	# Implement epsilon greedy action policy
	if np.random.random() < epsilon:
		return np.random.randint(0,len(values))
	else:
		return np.argmax(values)

def evaluation(env, Q_table, step_bound = 100, num_itr = 10):
	"""
	Semi-greedy evaluation for discrete state and discrete action spaces and an episodic environment.

	Input:
		env : an environment object. 
		Q : A numpy array. Q values for all state and action pairs. 
			Q.shape = (the number of states, the number of actions)
		step_bound : the maximum number of steps for each iteration
		num_itr : the number of iterations

	Output:
		Total number of steps taken to finish an episode (averaged over num_itr trials)
		Cumulative reward in an episode (averaged over num_itr trials)

	"""
	total_step = 0 
	total_reward = 0 
	itr = 0 
	while(itr<num_itr):
		step = 0
		np.random.seed()
		state = env.reset()
		reward = 0.0
		done = False
		while((not done) and (step < step_bound)):
			action = get_action_egreedy(Q_table[state], 0.05)
			r, state_n, done = env.step(state,action)
			state = state_n
			reward += r
			step +=1
		total_reward += reward
		total_step += step
		itr += 1
	return total_step/float(num_itr), total_reward/float(num_itr)
	