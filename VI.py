from maze import *
from value_plot import *

discount=0.9
iters=1000

def calcQ( state, action):
	Vsum = 0
	rllist = []
	slip_action = ACTMAP[action]
	slip_reward, slip_next_state, _ = env.step(state,slip_action, slip=False)
	rllist.append((slip_reward, slip_next_state, env.slip))
	reward, next_state, _ = env.step(state, action, slip=False)
	rllist.append((reward, next_state, 1-env.slip))
	for reward, next_state, pi in rllist:
		Vsum += pi * (reward + discount * values[next_state])
	return Vsum

env = Maze()
values = np.zeros(env.snum)
Qval = np.zeros((env.snum, env.anum))
optpolicies = np.zeros(env.snum)

# VI
for i in range(iters):
	tmpV = np.zeros(env.snum)
	for state in range(env.snum):
		if env.idx2cell[int(state/8)] == env.goal_pos:
			continue
		Vmax = float('-inf')
		for action in range(env.anum):
			Vsum = calcQ(state, action)
			Vmax = max(Vmax, Vsum)
		tmpV[state] = Vmax
	values = np.copy(tmpV)

for state in range(env.snum):
	for action in range(env.anum):
		Qval[state, action] = calcQ(state, action)

for state in range(env.snum):
	optpolicies[state] = np.argmax(Qval[state,:])

np.save('Results/Q_Values',Qval)
print Qval
#np.save('Optimal_policies',optpolicies)
print "Optimal Policies --> 0 - UP; 1 - DOWN; 2 - LEFT; 3 - RIGHT"
print optpolicies
value_plot(Qval, env)
