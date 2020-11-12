
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx 


def update_dynamics(G, contact_rate, removal_rate, dt=1):
	"""
	Updates the dynamics of the graph G (w/o updating the state of the graph).

	Args:
		G: graph modeling the community
		contact_rate: contact rate
		removal_rate: recovery/removal rate
		dt: time step

	Returns:
		Updated graph.
	"""

	for node, node_data in G.nodes(data=True):
		# Nodes have spent dt more time in their states
		node_data["duration"] += dt

		# Susceptible -> Infected
		if node_data["state"] == "S":
			# Get infectious and non-quarantined neighbors
			adj_infected = [
				n for n in G.adj[node]
				if G.nodes[n]["state"] == "I" \
				   and not G.nodes[n]["quarantined"]
			]
			# If one or more transmissions happen, the node gets infected
			p_infection = 1 - (1 - dt*contact_rate) ** len(adj_infected)
			infected = 1 == np.random.binomial(1, p_infection)
			if infected:
				node_data["next_state"] = "I"

		# Infected -> Removed
		if node_data["state"] == "I":
			if node_data["duration"] >= 1 / removal_rate:
				node_data["next_state"] = "R"

	return G


def update_tests(G, tested_nodes, current_time, resample=False):
	"""
	Update tests (retest testing subjects).

	Args:
		G: graph of community.
		tested_nodes: nodes from G to be tested.
		current_time: current time.
		resample: resample test subjects randomly every testing round.

	Returns:
		Updated graph, and a flag indicating if a test subject was infected.
	"""

	if resample:
		tested_nodes = np.random.choice(G.nodes, len(tested_nodes))

	tested_positive = False
	for test_node in tested_nodes:
		G.nodes[test_node]["last_test"] = current_time
		infectious = G.nodes[test_node]["state"] == "I"
		G.nodes[test_node]["quarantined"] = infectious
		tested_positive = tested_positive or infectious

	return G, tested_positive


def update_state(G, SIR, current_time, quarantine_length):
	"""
	Update the state/attributes of network G.

	Args:
		G: graph of community.
		SIR: a dict of compartment sizes indexed by the
			 compartment's name (e.g. "S" for susceptible).
		current_time: current time.
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
			quaran_time = current_time - node_data["last_test"]
			node_data["quarantined"] = quaran_time < quarantine_length

	return G, SIR


def outbreak_simulation(G,
					    dt=1,
					    initial_infected=1,
					    contact_rate=0.1,
					    removal_rate=0.2,
					    testing_capacity=0.01,
					    testing_interval=1,
					    testing_strategy="random",
					    quarantine_length=0,
					    report_interval=1000,
					    testing_mode=False,
					    reproduce=True):
	"""
	Simulates the spread of an infectious disease in a community modeled
	by the graph G.

	Args:
		G: graph modeling the community.
		dt: time step.
		initial_infected: number of infected nodes in the beginning.
		contact_rate: transmission rate (in units of 1/dt).
		removal_rate: recovery/removal rate (in units of 1/dt).
		testing_capacity: max tests possible every `testing_interval`,
		                  measured as a fraction of the whole population.
		testing_interval: testing interval (in units of dt).
		testing_strategy: how to sample test subjects from population.
		quarantine_length: time should be spent in quarantine.
		report_interval: reporting interval for SIR values (in units of dt).
		testing_mode: stops simulation when disease reaches a test subject.

	Returns:
		tuple containing S, I, R values of all time steps.
	"""

	SIR_record = []
	current_time = 0

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
		num_tests = round(testing_capacity * len(G.nodes))
		if testing_strategy == "random":
			tested_nodes = np.random.choice(G.nodes, num_tests)
		else:
			raise NotImplementedError(
				f"Testing strategy '{testing_strategy}' not implemented.")


	# Track the number of nodes in each compartment
	SIR = {
		"S": len(G.nodes) - len(infected_nodes),
		"I": len(infected_nodes),
		"R": 0,
	}

	# Initialize default attributes of nodes (i.e. people):
	# - state: the node's state at `current_time`
	# - next_state: the node's state at `current_time+dt`, if any
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
		G = update_dynamics(G, contact_rate, removal_rate, dt)
		current_time += dt

		# Retest every testing_interval
		if round(current_time / dt) % testing_interval == 0:
			G, tested_positive = update_tests(G, tested_nodes, current_time)

		# Update state of network
		G, SIR = update_state(G, SIR, current_time, quarantine_length)
		SIR_record.append(tuple(SIR.values()))

		# Show last SIR values
		if round(current_time / dt) % report_interval == 0:
			S, I, R = SIR_record[-1]
			print(f"current_time = {current_time}, S = {S}, I = {I}, R = {R}")

	return tuple(zip(*SIR_record))


def plot_infections(num_lines=5, max_t=30):
	"""
	Plot the average SIR curves of `num_lines` curves,
	and show each infection curve on the plot too.

	Args:
		num_lines: number of simulations to average.
		max_t: plot curves up to `max_t` days.
	"""
	N = 4000
	Q = 40
	P = 100000
	G = nx.barabasi_albert_graph(N, 3)

	lines_shape = (num_lines, max_t+1)
	S_lines = np.empty(lines_shape)
	I_lines = np.empty(lines_shape)
	R_lines = np.empty(lines_shape)

	for i in range(num_lines):
		S, I, R = outbreak_simulation(G, testing_interval=P)
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
		"initial_infected": 1,
		"contact_rate": 0.1,
		"removal_rate": 0.2,
		"testing_capacity": 0.01,
		"testing_interval": 1,
		"testing_strategy": "random",
		"quarantine_length": 0,
		"report_interval": 1,
		"testing_mode": False,
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

