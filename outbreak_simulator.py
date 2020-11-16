
import random
import itertools
import functools
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import gamma
from multiprocessing import Pool, Process


def chunk_generator(array, num_chunks, repeat=True):
	"""
	Generates a cycle of `num_chunks` chunks from `array`.
	if repeat is False, generates one cycle only.
	"""
	chunk_len = int(np.ceil(len(array) / num_chunks))
	array_iter = iter(array)
	while True:
		subset = tuple(itertools.islice(array_iter, chunk_len))
		if len(subset) > 0:
			yield subset
		elif repeat:
			array_iter = iter(array)
		else:
			return


def ffill(arr):
	"""Forward fill an array."""
	mask = np.isnan(arr)
	idx = np.where(~mask, np.arange(mask.shape[1]), 0)
	np.maximum.accumulate(idx, axis=1, out=idx)
	out = arr[np.arange(idx.shape[0])[:,None], idx]
	return out


class InfectionRate:
	# TODO: implement awareness.
	def __init__(self, rate, stochastic=False):
		self.rate = rate
		self.stochastic = stochastic

		# Use gamma distribution for infection rate
		gamma_pdf = lambda t: gamma.pdf(t, a=2.25, scale=2.8)
		# Store values up to 50 days (unlikely to go past that)
		gamma_cache = [gamma_pdf(t) for t in range(50)]
		# Renormalize s.t. peak == rate
		c = self.rate / max(gamma_cache)
		gamma_cache = [c * g for g in gamma_cache]

		self.gamma = gamma_cache

	def __call__(self, duration=None):
		return self.gamma[duration] if self.stochastic else self.rate


class RecoveryRate:
	def __init__(self, recovery_time=14, post_recovery_rate=1):
		self.recovery_time = recovery_time
		self.post_recovery_rate = post_recovery_rate

	def __call__(self, duration=None):
		if duration < self.recovery_time:
			return 1e-10  # effectively 0, just to avoid ZeroDivisionError
		else:
			return self.post_recovery_rate


class TestingPool:
	def __init__(self, tested_nodes, rounds=1, schedule=[True]):

		# Initialize a chunk / partition generator
		self.tested_chunks = chunk_generator(tested_nodes, rounds)

		# Default test schedule is everyday
		# Assuming `t` == 1 is the first weekday (i.e. Monday)
		# Example of a schedule where we test everyday except weekends:
		#     [True, True, True, True, True, False, False]
		self.t = 1
		self.schedule = schedule

	def next_round(self):
		"""
		Get next round of testing subjects.
		"""
		testing_round = []
		if self.schedule[ (self.t - 1) % len(self.schedule) ]:
			testing_round = next(self.tested_chunks)
		self.t += 1
		return testing_round


def update_dynamics(G, infection_rate, recovery_rate, dt=1):
	"""
	Updates the dynamics of the graph G (w/o updating the state of the graph).

	Args:
		G: graph modeling the community
		infection_rate: contact rate
		recovery_rate: recovery/removal rate
		dt: time step

	Returns:
		Updated graph.
	"""

	for node, node_data in G.nodes(data=True):
		# Nodes have spent dt more time in their states
		node_data["duration"] += dt

		# Susceptible -> Infected
		if node_data["state"] == "S":
			# Check infected and not quarantined neighbors
			for adj in G.adj[node]:
				infected = G.nodes[adj]["state"] == "I"
				quarantined = G.nodes[adj]["quarantined"]
				if infected and not quarantined:
					# Sample the daily probability of infecting a neighbor
					b = infection_rate(G.nodes[adj]["duration"])
					if np.random.binomial(int(dt), b) > 0:
						node_data["next_state"] = "I"
						break

		# Infected -> Removed
		if node_data["state"] == "I":
			# Sample the daily probability of recovering
			r = recovery_rate(node_data["duration"])
			if np.random.binomial(int(dt), r) > 0:
				node_data["next_state"] = "R"

	return G


def update_tests(G, t, tested_nodes, sensitivity=1., specificity=1.):
	"""
	Update tests (retest testing subjects).

	Args:
		G: graph of community.
		t: current time.
		tested_nodes: nodes from G to be tested.
		sensitivity: true positives / (true positives + false negatives).
		specificity: true negatives / (true negatives + false positives).

	Returns:
		Updated graph, and a flag whether a test subject was infected.
	"""

	for test_node in tested_nodes:
		G.nodes[test_node]["last_test"] = t
		if G.nodes[test_node]["state"] == "I":
			# Patient is positive, true positive rate is sensitivity
			test_positive = 1 == np.random.binomial(1, sensitivity)
		else:
			# Patient is negative, false positive rate is 1 - specificity
			test_positive = 1 == np.random.binomial(1, 1 - specificity)
		
		# Add patient to quarantine if tested positive
		# TODO: implement delay testing
		G.nodes[test_node]["quarantined"] = test_positive

	return G


def update_state(G, SIR, t, quarantine_length):
	"""
	Update the state/attributes of network G.

	Args:
		G: graph of community.
		SIR: a dict of compartment sizes indexed by the
			 compartment's name (e.g. "S" for susceptible).
		t: current time.
		quarantine_length: minimum time to quarantine

	Returns:
		Updated graph and new compartment sizes.
	"""
	for _, node_data in G.nodes(data=True):
		if node_data["next_state"] is not None:
			# Update compartment sizes
			SIR[node_data["state"]] -= 1
			SIR[node_data["next_state"]] += 1
			# Update state and reset state dynamics
			node_data["state"] = node_data["next_state"]
			node_data["next_state"] = None
			node_data["duration"] = 0

		# Check and update quarantine state
		if node_data["quarantined"]:
			quaran_time = t - node_data["last_test"]
			node_data["quarantined"] = quaran_time < quarantine_length

	return G, SIR


def outbreak_simulation(G,
					    dt=1,
					    initial_infected=1,
					    infection_rate=InfectionRate(0.1, True),
					    recovery_rate=RecoveryRate(5, 1/3),
					    testing_capacity=0.1,
					    testing_rounds=10,
					    testing_schedule=[True],
					    quarantine_length=14,
					    report_interval=1000,
					    stop_if_positive=False,):
	"""
	Simulates the spread of an infectious disease in a community modeled
	by the graph G.

	Args:
		G: graph modeling the community.
		dt: time step.
		initial_infected: number of infected nodes in the beginning.
		infection_rate: infection rate (in units of 1/dt).
		recovery_rate: recovery/removal rate (I -> R rate).
		testing_capacity: max tests possible every `testing_interval`,
		                  measured as a fraction of the whole population.
		testing_rounds: how many rounds to divide the testing pool.
		testing_schedule: a cyclical schedule where False means no testing.
		quarantine_length: time should be spent in quarantine.
		report_interval: reporting interval for SIR values (in dt).
		stop_if_positive: stops simulation when a test subject gets infected.

	Returns:
		tuple containing S, I, R values of all time steps.
	"""

	SIR_record = []
	t = 0

	# Infect some nodes randomly from population
	infected_nodes = np.random.choice(G.nodes, initial_infected)
	# Sample test subjects from population
	num_tested_nodes = round(testing_capacity * len(G.nodes))
	tested_nodes = np.random.choice(G.nodes, num_tested_nodes)
	# Create testing pool
	testing_pool = TestingPool(tested_nodes,
							   rounds=testing_rounds,
							   schedule=testing_schedule)


	# Track the number of nodes in each compartment
	SIR = {
		"S": len(G.nodes) - len(infected_nodes),
		"I": len(infected_nodes),
		"R": 0,
	}

	# Initialize default attributes of nodes (i.e. people):
	# - state: the node's state at `t`
	# - next_state: the node's state at `t+dt`, if any
	# - duration: the amount of time the node spent in the current state
	# - last_test: the time of the last test
	# - quarantined: whether the node is currently quarantined or not
	for _, node_data in G.nodes(data=True):
		node_data["state"] = "S"
		node_data["next_state"] =  None
		node_data["duration"] =  0
		node_data["last_test"] = None
		node_data["quarantined"] =  False
	for infected_node in infected_nodes:
		# Correct state of infected nodes
		G.nodes[infected_node]["state"] = "I"

	# Define stopping criterion based on simulation mode
	def should_stop():
		no_infected = SIR["I"] == 0
		positive_test = any(G.nodes[n]["state"] == "I" for n in tested_nodes)
		return no_infected or (stop_if_positive and positive_test)

	### Start simulation of network ###
	while not should_stop():

		# Update dynamics of network
		G = update_dynamics(G, infection_rate, recovery_rate, dt)
		t += dt

		# Test next round of test subjects
		testing_round = testing_pool.next_round()
		G = update_tests(G, t, testing_round)

		# Update state of network
		G, SIR = update_state(G, SIR, t, quarantine_length)
		SIR_record.append(tuple(SIR.values()))

		# Show last SIR values
		if round(t / dt) % report_interval == 0:
			S, I, R = SIR_record[-1]
			print(f"time = {t}, S = {S}, I = {I}, R = {R}")

	return tuple(zip(*SIR_record))


def plot_SIR(S, I, R):
	"""Create a simple plot of the SIR curve"""
	plt.figure()
	plt.plot(S, label="S")
	plt.plot(I, label="I")
	plt.plot(R, label="R")
	plt.legend()
	plt.show()


def plot_averaged_SIRs(SIRs, max_t=50, I_lines_only=True, I_mean_only=True):
	"""
	Plot SIR curves and their average.
	and show each infection curve on the plot too.

	Args:
		SIRs: a list of SIR curves.
		max_t: plot curves up to `max_t` days.
		I_lines_only: whether to plot the lines of I or all of S, I, and R.
		I_mean_only: whether to plot the mean of I or all of S, I, and R.
	"""

	S_color = u'#1f77b4'
	I_color = u'#ff7f0e'
	R_color = u'#2ca02c'

	lines_shape = (len(SIRs), max_t+1)
	S_lines = np.ones(lines_shape) * np.nan
	I_lines = np.ones(lines_shape) * np.nan
	R_lines = np.ones(lines_shape) * np.nan

	# Create multi-array of all SIR curves up to max_t
	for i, SIR in enumerate(SIRs):
		S, I, R = SIR
		S, I, R = np.array(S), np.array(I), np.array(R)
		S_lines[i, :S.shape[0]] = S[:max_t+1]
		I_lines[i, :I.shape[0]] = I[:max_t+1]
		R_lines[i, :R.shape[0]] = R[:max_t+1]

	# Forward fill final values from simulation
	S_lines = ffill(S_lines)
	I_lines = ffill(I_lines)
	R_lines = ffill(R_lines)

	# Plot the averages of S, I, and R curves
	fig = plt.figure(figsize=(13, 8))
	plt.plot(I_lines.mean(0), label="I", color=I_color, linewidth=3)
	if not I_mean_only:
		plt.plot(S_lines.mean(0), label="S", color=S_color, linewidth=3)
		plt.plot(R_lines.mean(0), label="R", color=R_color, linewidth=3)

	# Plot all I curves to visualize simulation runs
	[plt.plot(I, color=I_color, linewidth=0.5) for I in I_lines]
	if not I_lines_only:
		[plt.plot(S, color=S_color, linewidth=0.5) for S in S_lines]
		[plt.plot(R, color=R_color, linewidth=0.5) for R in R_lines]

	# Configure plot, show, and save
	plt.legend()
	plt.grid(which="major")
	plt.xlim(0, max_t)
	plt.show()
	fig.savefig("SIR.png")


def repeat_simulation(sim_config={},
					  num_sim=100,
					  init_G=None,
					  regenerate=False,
					  processes=None):
	"""
	Repeats an outbreak simulation given its config.
	Runs in parallel.

	Args:
		sim_config: config of the outbreak simulation.
		num_sim: number of simulations to run.
		init_G: a function that generates / initializes a graph.
		regenerate: if True, regenerate the graph for every simulation.
		processes: number of processes to run in parallel.

	Return:
		list of SIR curves for all simulations.
	"""

	# Define default graph initializer
	if init_G is None:
		init_G = lambda: nx.barabasi_albert_graph(4000, 3)
	
	# Define simulation task based on given config.
	# This will return a function `sim(G)` that takes a graph as an arg.
	sim = functools.partial(outbreak_simulation, **sim_config)

	# Run simulations in parallel
	with Pool(processes=processes) as pool:
		if regenerate:
			graphs = [init_G() for _ in range(num_sim)]
		else:
			G = init_G()
			graphs = [G] * num_sim
		SIRs = pool.map(sim, graphs)

	return SIRs


def simulation_example():
	G = nx.barabasi_albert_graph(4000, 3)
	plot_SIR(*outbreak_simulation(G))


def repeat_simulation_example():
	plot_averaged_SIRs(repeat_simulation())


def main():
	random.seed(123)
	np.random.seed(123)
	repeat_simulation_example()


if __name__ == '__main__':
	main()

