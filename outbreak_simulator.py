
import os
import logging
import random
import itertools
import functools
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import gamma
from multiprocessing import Pool, Process

FORMAT = "%(name)s.%(process)d.%(levelname)s %(message)s"
logging.basicConfig(filename="sim.log", filemode="w", level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger(__name__)


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
	def __init__(self, infection_rate, gamma_infection=False):
		self.infection_rate = infection_rate
		self.gamma_infection = gamma_infection

		# Use gamma distribution for infection rate
		gamma_pdf = lambda t: gamma.pdf(t, a=2.25, scale=2.8)
		# Store values up to 50 days (unlikely to go past that)
		gamma_cache = [gamma_pdf(t) for t in range(51)]
		# Renormalize s.t. peak == rate
		c = self.infection_rate / max(gamma_cache)
		gamma_cache = [c * g for g in gamma_cache]

		self.gamma = gamma_cache

	def __call__(self, duration=None):
		if self.gamma_infection:
			return self.gamma[duration]
		else:
			return self.infection_rate


class RecoveryRate:
	"""
	Patient has to spend `recovery_time` days first and then they
	will heal at a rate of `recovery_rate`.
	"""
	def __init__(self, recovery_time=14, recovery_rate=1):
		self.recovery_time = recovery_time
		self.recovery_rate = recovery_rate

	def __call__(self, duration=None):
		if duration < self.recovery_time:
			return 1e-10  # effectively 0, just to avoid ZeroDivisionError
		else:
			return self.recovery_rate


class TestingPool:
	def __init__(self, tested_nodes, rounds=1, schedule=[True]):

		# Initialize a chunk / partition generator
		self.tested_chunks = chunk_generator(tested_nodes, rounds)

		# Default test schedule is everyday
		# Assuming `t` == 1 is the first weekday (i.e. Monday)
		# Example of a schedule where we test everyday except weekends:
		#     [True, True, True, True, True, False, False]
		self.t = 1
		self.schedule = schedule if len(schedule) > 0 else [True]

	def next_round(self):
		"""
		Get next round of testing subjects.
		"""
		testing_round = []
		if self.schedule[ (self.t - 1) % len(self.schedule) ]:
			testing_round = next(self.tested_chunks)
		self.t += 1
		return testing_round


def update_dynamics(G, t, infection_rate, recovery_rate, dt=1):
	"""
	Updates the dynamics of the graph G (w/o updating the state of the graph).

	Args:
		G: graph modeling the community
		t: current time.
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
				quarantined = G.nodes[adj]["quarantine_rem"] >= 0
				if infected and not quarantined:
					# Sample the daily probability of infecting a neighbor
					b = infection_rate(G.nodes[adj]["duration"])
					if np.random.binomial(int(dt), b) > 0:
						node_data["next_state"] = "I"
						logger.debug(f"[{t}] Node #{adj} infected node #{node}")
						break

		# Infected -> Removed
		if node_data["state"] == "I":
			# Sample the daily probability of recovering
			r = recovery_rate(node_data["duration"])
			if np.random.binomial(int(dt), r) > 0:
				node_data["next_state"] = "R"
				logger.debug(f"[{t}] Node #{node} recovered")

	return G


def test_state(node_state, sensitivity=1., specificity=1.,):
	"""
	Tests node given test sensitivity and specificity.

	Args:
		node_state: state of node (S, I, or R).
		sensitivity: true positives / (true positives + false negatives).
		specificity: true negatives / (true negatives + false positives).

	Returns:
		Result of test (True if positive, False if negative).
	"""
	if node_state == "I":
		# Patient is positive, true positive rate is sensitivity
		positive = 1 == np.random.binomial(1, sensitivity)
	else:
		# Patient is negative, false positive rate is 1 - specificity
		positive = 1 == np.random.binomial(1, 1 - specificity)

	return positive


def update_tests(G, t, tested_nodes,
				 quarantine_length=14,
	             test_sensitivity=1.,
	             test_specificity=1.,
	             delay=0):
	"""
	Update tests (retest testing subjects).

	Args:
		G: graph of community.
		t: current time.
		tested_nodes: nodes from G to be tested.
		quarantine_length: time length of quarantine.
		test_sensitivity: test sensitivity.
		test_specificity: test specificity.
		delay: delay of test result (0 for immediate test result). TODO

	Returns:
		Updated graph.
	"""

	# If quarantine length is 0 or less, then testing is useless
	if quarantine_length <= 0:
		return G

	for tested_node in tested_nodes:
		# Skip if already recovered XXX: why?
		if G.nodes[tested_node]["state"] == "R":
			pass #continue
		# Skip if still in quarantine
		if G.nodes[tested_node]["quarantine_rem"] >= 0:
			continue

		# Test node and record test time
		positive = test_state(G.nodes[tested_node]["state"],
							  test_sensitivity, test_specificity)

		# Add patient to quarantine if tested positive
		if positive and quarantine_length > 0:
			logger.debug(f"[{t}] Quarantining positive node #{tested_node}")
			G.nodes[tested_node]["quarantine_rem"] = quarantine_length

	return G


def update_state(G, SIR, t, 
				 quarantine_length=14,
				 test_sensitivity=1.,
	             test_specificity=1.,):
	"""
	Update the state/attributes of network G.

	Args:
		G: graph of community.
		SIR: a dict of compartment sizes indexed by the
			 compartment's name (e.g. "S" for susceptible).
		t: current time.
		quarantine_length: minimum time to quarantine
		test_sensitivity: test sensitivity.
		test_specificity: test specificity.

	Returns:
		Updated graph and new compartment sizes.
	"""
	for _, node_data in G.nodes(data=True):
		# Check quarantine state
		if node_data["quarantine_rem"] > 0:
			node_data["quarantine_rem"] -= 1
		elif node_data["quarantine_rem"] == 0:
			# Node finished quarantine, retest
			positive = test_state(node_data["state"],
								  test_sensitivity, test_specificity)
			# Increase quarantine by 1 week if still positive
			if positive:
				node_data["quarantine_rem"] = 7  # XXX: hardcoded values
			else:
				node_data["quarantine_rem"] = -1  # not quarantined

		if node_data["next_state"] is not None:
			# Update compartment sizes
			SIR[node_data["state"]] -= 1
			SIR[node_data["next_state"]] += 1
			# Update state and reset state dynamics
			node_data["state"] = node_data["next_state"]
			node_data["next_state"] = None
			node_data["duration"] = 0

	return G, SIR


def outbreak_simulation(G,
						dt=1,
						initial_infected=1,
						infection_rate=InfectionRate(0.1, False),
						recovery_rate=RecoveryRate(5, 1/3),
						testing_capacity=0.1,
						testing_rounds=10,
						testing_schedule=[True],
						test_sensitivity=1.0,
						test_specificity=1.0,
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
		test_sensitivity: test sensitivity.
		test_specificity: test specificity.
		quarantine_length: time should be spent in quarantine.
		report_interval: reporting interval for SIR values (in dt).
		stop_if_positive: stops simulation when a test subject gets infected.

	Returns:
		tuple containing S, I, R values of all time steps.
	"""

	SIR_record = []
	t = 0

	# Infect some nodes randomly from population
	infected_nodes = np.random.choice(G.nodes, initial_infected, replace=False)
	# Sample test subjects from population
	num_tested_nodes = round(testing_capacity * len(G.nodes))
	tested_nodes = np.random.choice(G.nodes, num_tested_nodes, replace=False)
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
	# - state: node's state at `t`
	# - next_state: node's state at `t+dt`, None if not changed
	# - duration: amount of time the node spent in the current state
	# - quarantine_rem: remaining time of quarantine (< 0 means not in qrtn)
	for _, node_data in G.nodes(data=True):
		node_data["state"] = "S"
		node_data["next_state"] =  None
		node_data["duration"] =  0
		node_data["quarantine_rem"] =  -1
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
		t += dt
		G = update_dynamics(G, t, infection_rate, recovery_rate, dt)

		# Test next round of test subjects
		G = update_tests(G, t, testing_pool.next_round(),
						 quarantine_length=quarantine_length,
						 test_sensitivity=test_sensitivity,
						 test_specificity=test_specificity)

		# Update state of network
		G, SIR = update_state(G, SIR, t,
						 	  quarantine_length=quarantine_length,
						 	  test_sensitivity=test_sensitivity,
						 	  test_specificity=test_specificity)
		
		# Record SIR values
		SIR_record.append(tuple(SIR.values()))

		# Show last SIR values
		if round(t / dt) % report_interval == 0:
			S, I, R = SIR_record[-1]
			logger.info(f"[{t}] S = {S}, I = {I}, R = {R}")

	return tuple(zip(*SIR_record))


def plot_SIR(S, I, R):
	"""Create a simple plot of the SIR curve"""
	plt.figure()
	plt.plot(S, label="S")
	plt.plot(I, label="I")
	plt.plot(R, label="R")
	plt.legend()
	plt.show()


def plot_averaged_SIRs(SIRs,
					   max_t="auto",
					   lines_to_plot="IR",
					   means_to_plot="SIR",
					   figname="SIRs.png",
					   show_plot=False):
	"""
	Plot SIR curves and their average.
	and show each infection curve on the plot too.

	Args:
		SIRs: a list of SIR curves.
		max_t: plot up to `max_t` days, set to 'auto' to auto-detect max.
		lines_to_plot: plot the lines of all sims for each comp. in `lines`.
		means_to_plot: plot the mean of all sims for each comp. in `means`.
		figname: name of figure to save, None if no need to save fig.
		show_plot: show plot if True.
	"""

	compartments = ("S", "I", "R")
	colors = {"S": u'#1f77b4', "I": u'#ff7f0e', "R": u'#2ca02c'}

	if max_t == "auto":
		max_t = max(len(line) for SIR in SIRs for line in SIR)

	lines_shape = (len(SIRs), max_t+1)
	S_lines = np.zeros(lines_shape) + np.nan
	I_lines = np.zeros(lines_shape) + np.nan
	R_lines = np.zeros(lines_shape) + np.nan

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

	# Pack lines in a dict
	lines = {"S": S_lines, "I": I_lines, "R": R_lines}

	# Plot the averages of S, I, and R curves
	fig = plt.figure(figsize=(13, 8))
	for comp in compartments:
		if comp in means_to_plot:
			plt.plot(lines[comp].mean(0),
					 label=comp, color=colors[comp], linewidth=3)

	# Plot all I curves to visualize simulation runs
	for comp in compartments:
		if comp in lines_to_plot:
			for comp_line in lines[comp]:
				plt.plot(comp_line, color=colors[comp], linewidth=0.5)

	# Configure plot, show, and save
	plt.legend()
	plt.grid(which="major")
	plt.title(f"SIR Curves of {len(SIRs)} Simulations")
	#plt.xlim(0, max_t)
	if show_plot:
		plt.show()
	if figname is not None:
		fig.savefig(figname)


def repeat_simulation(G=nx.barabasi_albert_graph(4000, 3),
					  sim_config={},
					  num_sim=100,
					  regenerate_graph=False,
					  parallel=None):
	"""
	Repeats an outbreak simulation given its config.

	Args:
		sim_config: config of the outbreak simulation.
		num_sim: number of simulations to run.
		G: can be a graph or a callable that generates a graph.
		parallel: number of processes to run in parallel (# of CPUs if 0).
				   Doesn't run in parallel if None.

	Return:
		list of SIR curves for all simulations.
	"""
	
	# Define simulation task based on given config.
	# This will return a function `sim(G)` that takes a graph as an arg.
	sim = functools.partial(outbreak_simulation, **sim_config)

	# initialize graphs for each simulation
	new_G = G if callable(G) else lambda: G.copy()
	graphs = (new_G() for _ in range(num_sim))

	# Run simulations in parallel
	if parallel is None or num_sim <= 10:
		SIRs = [sim(graph) for graph in graphs]
	else:
		processes = parallel if parallel > 0 else None
		with Pool(processes=processes) as pool:
			SIRs = pool.map(sim, graphs)


	return SIRs


def main():
	random.seed(123)
	np.random.seed(123)
	plot_averaged_SIRs(repeat_simulation(), figname=None, show_plot=True)


if __name__ == '__main__':
	main()

