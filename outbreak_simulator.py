
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import gamma


class InfectionRate:
	def __init__(self, rate, stochastic=False):
		self.rate = rate
		self.stochastic = stochastic

		if not stochastic:
			# Constant function
			self._get_rate = lambda _: rate
		else:
			# Use gamma distribution for infection rate
			gamma_pdf = lambda t: gamma.pdf(t, a=2.25, scale=2.8)
			# Store values up to 50 days (unlikely to go past that)
			gamma_cache = [gamma_pdf(t) for t in range(50)]
			# Renormalize s.t. peak == rate
			c = rate / max(gamma_cache)
			gamma_cache = [c * g for g in gamma_cache]
			self._get_rate = lambda t: gamma_cache[t]

	def __call__(self, duration=None):
		return self._get_rate(duration)


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
	def __init__(self, tested_nodes, num_chunks=1, schedule=[True]):

		# Initialize a chunk / partition generator
		self.tested_chunks = self._chunk_generator(tested_nodes, num_chunks)

		# Default test schedule is everyday
		# Assuming `t` == 1 is the first weekday (i.e. Monday)
		# Example of a schedule where we test everyday except weekends:
		#     [True, True, True, True, True, False, False]
		self.t = 1
		self.schedule = schedule

	def _chunk_generator(self, tested_nodes, num_chunks):
		"""
		Generates a cycle of `num_chunks` chunks from `tested_nodes`.
		"""
		chunk_len = int(np.ceil(len(tested_nodes) / num_chunks))
		while True:
			for i in range(0, len(tested_nodes), chunk_len):
				yield tested_nodes[i:i+chunk_len]

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


def update_tests(G, t, tested_nodes):
	"""
	Update tests (retest testing subjects).

	Args:
		G: graph of community.
		t: current time.
		tested_nodes: nodes from G to be tested.
		resample: resample test subjects randomly every testing round.

	Returns:
		Updated graph, and a flag whether a test subject was infected.
	"""

	tested_positive = False
	for test_node in tested_nodes:
		G.nodes[test_node]["last_test"] = t
		infectious = G.nodes[test_node]["state"] == "I"
		G.nodes[test_node]["quarantined"] = infectious
		tested_positive = tested_positive or infectious

	return G, tested_positive


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
					    testing_capacity=0.01,
					    testing_rounds=1,
					    testing_interval=1,
					    quarantine_length=14,
					    report_interval=1000,
					    testing_mode=False,
					    reproduce=False):
	"""
	Simulates the spread of an infectious disease in a community modeled
	by the graph G.

	TODO: There is a bug where the number of infections becomes
	      very large in magnitude (could be an overflow bug?)

	Args:
		G: graph modeling the community.
		dt: time step.
		initial_infected: number of infected nodes in the beginning.
		infection_rate: infection rate (in units of 1/dt).
		recovery_rate: recovery/removal rate (I -> R rate).
		testing_capacity: max tests possible every `testing_interval`,
		                  measured as a fraction of the whole population.
		testing_rounds: how many rounds to divide the testing pool.
		testing_interval: testing interval (in units of dt).
		quarantine_length: time should be spent in quarantine.
		report_interval: reporting interval for SIR values (in dt).
		testing_mode: stops simulation when a test subject gets infected.

	Returns:
		tuple containing S, I, R values of all time steps.
	"""

	SIR_record = []
	t = 0

	 # XXX
	if reproduce:
		M = 800
		Q = round(testing_capacity * len(G.nodes))
		N_nodes = G.nodes
		tested_nodes = random.sample(N_nodes, Q)
		edges = random.sample(N_nodes,M)
		infected_nodes = random.sample(set(edges), 1)
	else:
		# Infect some nodes randomly from population
		infected_nodes = np.random.choice(G.nodes, initial_infected)
		# Sample test subjects from population
		num_tested_nodes = round(testing_capacity * len(G.nodes))
		tested_nodes = np.random.choice(G.nodes, num_tested_nodes)

	# Create testing pool
	testing_pool = TestingPool(tested_nodes, num_chunks=testing_rounds)


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
	tested_positive = False
	def should_stop():
		no_infected = SIR["I"] == 0
		stop_testing_mode = testing_mode and tested_positive
		return no_infected or stop_testing_mode

	### Start simulation of network ###
	while not should_stop():

		# Update dynamics of network
		G = update_dynamics(G, infection_rate, recovery_rate, dt)
		t += dt

		# Retest every testing_interval
		if round(t / dt) % testing_interval == 0:
			testing_round = testing_pool.next_round()
			G, tested_positive = update_tests(G, t, testing_round)

		# Update state of network
		G, SIR = update_state(G, SIR, t, quarantine_length)
		SIR_record.append(tuple(SIR.values()))

		# Show last SIR values
		if round(t / dt) % report_interval == 0:
			S, I, R = SIR_record[-1]
			print(f"time = {t}, S = {S}, I = {I}, R = {R}")

	return tuple(zip(*SIR_record))


def plot_infections(num_lines=20, max_t=30):
	"""
	Plot the average SIR curves of `num_lines` curves,
	and show each infection curve on the plot too.

	Args:
		num_lines: number of simulations to average.
		max_t: plot curves up to `max_t` days.
	"""
	N = 4000
	G = nx.barabasi_albert_graph(N, 3)

	Q = 400  # num of test subjects
	testing_capacity = Q / N
	testing_rounds = 10
	testing_interval = 1

	lines_shape = (num_lines, max_t+1)
	S_lines = np.empty(lines_shape)
	I_lines = np.empty(lines_shape)
	R_lines = np.empty(lines_shape)

	for i in range(num_lines):
		S, I, R = outbreak_simulation(G, testing_capacity=testing_capacity,
										 testing_rounds=testing_rounds,
										 testing_interval=testing_interval,)
		S, I, R = np.array(S), np.array(I), np.array(R)
		S_lines[i, :S.shape[0]] = S[:max_t+1]
		I_lines[i, :I.shape[0]] = I[:max_t+1]
		R_lines[i, :R.shape[0]] = R[:max_t+1]

	fig = plt.figure()
	S_mean = np.nanmean(np.array(S_lines), axis=0)
	I_mean = np.nanmean(np.array(I_lines), axis=0)
	R_mean = np.nanmean(np.array(R_lines), axis=0)

	#plt.plot(S_mean, label="S", color=u'#1f77b4', linewidth=2)
	plt.plot(I_mean, label="I", color=u'#ff7f0e', linewidth=2)
	#plt.plot(R_mean, label="R", color=u'#2ca02c', linewidth=2)

	for I in I_lines:
		plt.plot(I, color=u'#ff7f0e', linewidth=0.5)

	plt.legend()
	plt.grid(which="major")
	plt.xlim(0, max_t)
	plt.show()
	fig.savefig("SIR_curves.png")


def main():
	# Define network structure and outbreak configurations
	G = nx.barabasi_albert_graph(4000, 3)
	outbreak_config = {
		"quarantine_length": 0,
		"report_interval": 1,
		"testing_mode": True,
	}
	# Run the simulation
	S, I, R = outbreak_simulation(G, **outbreak_config)

	# Plot SIR curve
	plt.figure()
	plt.plot(S, label="S")
	plt.plot(I, label="I")
	plt.plot(R, label="R")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	
	random.seed(123)
	np.random.seed(123)

	plot_infections()

