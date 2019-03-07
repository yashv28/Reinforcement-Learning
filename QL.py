from maze import *
from evaluation import *
import matplotlib.pyplot as plt
import random

discount=0.9
alpha=0.2
exploration_factor=0.5
iters = 5000

def calcQ(state, action):
	if (state, action) not in Qval:
		Qval[(state, action)] = 0
	return Qval[(state, action)]

def calcV(state, aclist):
	Vmax = float('-inf')
	for action in aclist:
		qv = calcQ(state, action)
		Vmax = max(Vmax, qv)
	return Vmax

env = Maze()
aclist = np.arange(env.anum).tolist()
Qval = {}

eval_steps, eval_reward, RMSE = [], [], []
optQ = np.load('Results/Q_Values.npy')

# Q-learning
for i in range(iters):
	state = env.reset()
	done = False
	while not done:
		if np.random.random() < exploration_factor:
			optaction = np.random.choice(aclist)
		else:
			optaction = None
			Vmax = float('-inf')
			for action in aclist:
				qv = calcQ(state, action)
				if qv > Vmax:
					Vmax = qv
					optaction = action
				elif qv == Vmax:
					optaction = random.choice([optaction, action])
		reward, next_state, done = env.step(state, optaction)
		Qval[(state, optaction)] = (1-alpha)*calcQ(state, optaction) + alpha*(reward+(discount*calcV(next_state,aclist)))
		state = next_state

	tmpQ = np.zeros((env.snum, env.anum))
	for k, v in Qval.items():
		tmpQ[k] = v
	avg_step, avg_reward = evaluation(env, tmpQ)
	eval_steps.append(avg_step)
	eval_reward.append(avg_reward)
	RMSE.append(np.sqrt(np.mean((tmpQ-optQ)**2)))

fig1 = plt.figure(figsize=(10,5))
plt.plot(np.arange(iters), eval_steps)
plt.xlabel("Iterations")
plt.ylabel("steps")
plt.show(fig1)

fig2 = plt.figure(figsize=(10,5))
plt.plot(np.arange(iters), eval_reward)
plt.xlabel("Iterations")
plt.ylabel("reward")
plt.show(fig2)

fig3 = plt.figure(figsize=(10,5))
plt.plot(np.arange(iters), RMSE)
plt.xlabel("Iterations")
plt.ylabel("RMSE")
plt.show(fig3)
