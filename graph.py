def plot_cost_to_go_mountain_car(env, num_tiles=20):
	x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
	y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
	X, Y = np.meshgrid(x, y)
	Z = np.apply_along_axis(lambda _: -np.max(predict(_)), 2, np.dstack([X, Y]))

	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
						   cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
	ax.set_xlabel('Position')
	ax.set_ylabel('Velocity')
	ax.set_zlabel('Value')
	ax.set_title("MountainCar-v0 Predicted Value")
	fig.colorbar(surf)
	plt.show()

def plot_cost_to_go_acro(env, num_tiles=20):
	x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
	y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
	a = np.linspace(env.observation_space.low[2], env.observation_space.high[2], num=num_tiles)
	b = np.linspace(env.observation_space.low[3], env.observation_space.high[3], num=num_tiles)
	c = np.linspace(env.observation_space.low[4], env.observation_space.high[4], num=num_tiles)
	d = np.linspace(env.observation_space.low[5], env.observation_space.high[5], num=num_tiles)
	X, Y, A, B, C, D = np.meshgrid(x, y,a,b,c,d)
	Z = np.apply_along_axis(lambda _: -np.max(predict(_)), 6, np.dstack([X, Y, A, B, C, D]))

	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
						   cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
	ax.set_xlabel('Position')
	ax.set_ylabel('Velocity')
	ax.set_zlabel('Value')
	ax.set_title("MountainCar-v0 Predicted Value")
	fig.colorbar(surf)
	plt.show()

if mc:
	plot_cost_to_go_mountain_car(env)
else:
	plot_cost_to_go_acro(env)
