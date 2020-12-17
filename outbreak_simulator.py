
import os
import pickle
import logging
import random
import itertools
import functools
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import gamma
from multiprocessing import Pool, Process

LOG_FORMAT = "%(name)s.%(process)d.%(levelname)s: %(message)s"
logging.basicConfig(filename="sim.log", filemode="a", level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def init_random_seed(seed):
	random.seed(seed)
	np.random.seed(seed)


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
		# Store values up to 100 days (unlikely to go beyond that)
		gamma_cache = [gamma_pdf(t) for t in range(101)]
		# Renormalize s.t. peak == rate
		c = self.infection_rate / max(gamma_cache)
		gamma_cache = [c * g for g in gamma_cache]

		self.gamma = gamma_cache

	def __call__(self, duration=None):
		if self.gamma_infection:
			if duration < len(self.gamma):
				return self.gamma[duration]
			else:
				return 0
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
	def __init__(self, nodes, rounds=1, schedule=[True]):

		self.nodes = nodes
		self.rounds = rounds

		# Initialize a chunk / partition generator
		self._tested_chunks = chunk_generator(self.nodes, self.rounds)

		# Default test schedule is everyday
		# Assuming `day` == 0 is the first weekday (i.e. Monday)
		# Example of a schedule where we test everyday except weekends:
		#     [True, True, True, True, True, False, False]
		self.day = 0
		self.schedule = schedule if len(schedule) > 0 else [True]

	def next_round(self):
		"""
		Get next round of testing subjects.
		"""
		testing_round = []
		if self.schedule[self.day]:
			testing_round = next(self._tested_chunks)
		self.day = (self.day + 1) % len(self.schedule)
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
		# If node is quarantined, reduce remaining time in quarantine
		if node_data["q_state"]:
			node_data["q_rem"] -= dt

		# Susceptible -> Infected
		if node_data["state"] == "S":
			# Check infected and not quarantined neighbors
			for adj in G.adj[node]:
				infected = G.nodes[adj]["state"] == "I"
				quarantined = G.nodes[adj]["q_state"]
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


def update_tests(G, t, testing_pool,
				 q_len=14,
				 q_extend=7,
	             sensitivity=1.,
	             specificity=1.,
	             delay=0,):
	"""
	Update tests (retest testing subjects) and quarantine state.

	Note the entry and exit points of quarantine states (i.e. 'q_state'):
	• Entry only happens when a subject receives a positive test result.
	• Exit only happens when a quarantined subject passes quarantine time and
	  receives negative test result.
	
	In the first loop, we only go through the fraction of the testing pool and
	that are supposed to be testing. That fraction is determined by the
	number of testing rounds in the testing pool. This is the testing round.
	In the second loop, we go through _all_ test subjects and handle all
	remaining cases (subjects expecting test result, quarantined subjects,
	etc.)

	Args:
		G: graph of community.
		t: current time.
		testing_pool: testing pool containing all test subjects.
		q_len: time length of quarantine.
		q_extend: extension time if still positive after finishing quarantine
		sensitivity: test sensitivity.
		specificity: test specificity.
		delay: delay of test result (0 for immediate test result).
			   XXX: does not delay test results for quarantine extension.

	Returns:
		Updated graph.
	"""

	# If quarantine length is 0 or less, then testing is useless XXX: not always
	if q_len <= 0:
		return G

	# Resume the next testing round
	for node in testing_pool.next_round():
		# Skip quarantined nodes or nodes that are waiting for test results
		if G.nodes[node]["q_state"]:
			continue
		# Test node
		positive = test_state(G.nodes[node]["state"],
							  sensitivity, specificity)
		# If positive, record time to receive positive test result back
		if positive and G.nodes[node]["positive_t"] is None:
			G.nodes[node]["positive_t"] = t + delay

	# Update state of all subjects
	for node in testing_pool.nodes:
		# Handle nodes expecting positive test result
		if t == G.nodes[node]["positive_t"]:
			logger.debug(f"[{t}] Node #{node} enters quarantine")
			# Quarantine node
			G.nodes[node]["q_state"] = True
			G.nodes[node]["q_rem"] = q_len

		# Handle nodes that finished quarantine
		if G.nodes[node]["q_state"] and G.nodes[node]["q_rem"] <= 0:
			positive = test_state(G.nodes[node]["state"],
								  sensitivity, specificity)
			# Extend quarantine if still positive
			if positive:
				G.nodes[node]["q_rem"] = q_extend
				logger.debug(f"[{t}] Node #{node} extends quarantine")
			# Exit quarantine otherwise
			else:
				G.nodes[node]["q_state"] = False
				logger.debug(f"[{t}] Node #{node} exits quarantine")

	return G


def update_state(G, SIR, t):
	"""
	Update the state/attributes of network G.

	Args:
		G: graph of community.
		SIR: a dict of compartment sizes indexed by the
			 compartment's name (e.g. "S" for susceptible).
		t: current time.

	Returns:
		Updated graph and new compartment sizes.
	"""
	for node, node_data in G.nodes(data=True):
		if node_data["next_state"] is not None:
			dynamics = node_data["state"] +  " -> " + node_data["next_state"]
			duration = node_data["duration"]
			logger.debug(f"[{t}] Node #{node}: {dynamics} ({duration} days)")
			# Update compartment sizes
			SIR[node_data["state"]] -= 1
			SIR[node_data["next_state"]] += 1
			# Update state and reset state dynamics
			node_data["state"] = node_data["next_state"]
			node_data["next_state"] = None
			node_data["duration"] = 0

	return G, SIR


def outbreak_simulation(sim_id,
						random_seed=None,
						G=nx.barabasi_albert_graph(4000, 3),
						dt=1,
						initial_infected=1,
						infection_rate=InfectionRate(0.1, False),
						recovery_rate=RecoveryRate(5, 1/3),
						testing_capacity=0.1,
						testing_rounds=10,
						testing_schedule=[True],
						test_sensitivity=1.0,
						test_specificity=1.0,
						test_delay=0,
						quarantine_length=14,
						report_interval=1000,
						stop_if_positive=False,):
	"""
	Simulates the spread of an infectious disease in a community modeled
	by the graph G.

	The algorithm for the simulation goes as follows:
	1) Set t <- 0.
	2) Initialize nodes attributes in G:
		a) Sample infected nodes from G.
		b) Sample test subjects from G and initialize the testing pool.
		d) Initialize default/initial/t0 attributes of all nodes in G.
		e) Adjust attributes of the sampled nodes.
	3) Initialize initial SIR values at time step 0.
	4) Define stopping criterion.
	5) Loop while stopping criterion is not met:
		a) Update dynamics of nodes from t to t+dt.
		b) Set t <- t+dt.
		c) Update state of all nodes.
		d) Run a testing round on the testing pool.
		e) Save SIR values for time step t.
	6) Return SIR values for all time steps.

	Args:
		sim_id: a unique non-negative integer identifier
		random_seed: random seed.
		G: a graph or a graph generator, modeling the community.
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
		test_delay: amount of time needed to get results after testing
		quarantine_length: time should be spent in quarantine.
		report_interval: reporting interval for SIR values (in dt).
		stop_if_positive: stops simulation when a test subject gets infected.

	Returns:
		tuple containing S, I, R values of all time steps.
	"""

	SIR_record = []
	t = 0
	logger.info(f"Start simulation #{sim_id}.")
	logger.info(f"Experiment's random seed = {random_seed}.")

	# Initialize random seed, give unique seed for each simulation
	if random_seed is not None:
		init_random_seed(random_seed)
		for _ in range(sim_id + 1):
			sim_seed = np.random.randint(1 << 32)
		init_random_seed(sim_seed)
		logger.info(f"Simulation's random seed = {sim_seed}.")

	# Generate graph if given a graph generator
	G = G() if callable(G) else G.copy()

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
	# - state: node's state at t
	# - next_state: node's state at t+dt, None if same as t
	# - duration: days spent in the current state
	# - positive_t: time when positive test result is out
	# - q_state: True if in quarantine, False if not
	# - q_rem: days remaining in quarantine
	for _, node_data in G.nodes(data=True):
		node_data["state"] = "S"
		node_data["next_state"] =  None
		node_data["duration"] =  0
		node_data["positive_t"] = None
		node_data["q_state"] = False
		node_data["q_rem"] = 0
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
		G = update_dynamics(G, t, infection_rate, recovery_rate, dt)
		t += dt

		# Update state of network
		G, SIR = update_state(G, SIR, t)

		# Test next round of test subjects
		G = update_tests(G, t, testing_pool,
						 q_len=quarantine_length,
						 sensitivity=test_sensitivity,
						 specificity=test_specificity,
						 delay=test_delay)
		
		# Record SIR values
		SIR_record.append(tuple(SIR.values()))

		# Show last SIR values
		if round(t / dt) % report_interval == 0:
			S, I, R = SIR_record[-1]
			logger.info(f"[{t}] S = {S}, I = {I}, R = {R}")

	logger.info(f"[{t}] Simulation done.")
	S, I, R = SIR_record[-1]
	logger.info(f"[{t}] Cumulative infected individuals = {I+R}")

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
					   max_t=None,
					   lines_to_plot="IR",
					   means_to_plot="SIR",
					   figname="SIRs.png",
					   figtitle=None,
					   show_plot=False,
					   save_data=False):
	"""
	Plot SIR curves and their average.
	and show each infection curve on the plot too.

	Args:
		SIRs: a list of SIR curves.
		max_t: plot up to `max_t` days, set to None to auto-detect max.
		lines_to_plot: plot the lines of all sims for each comp. in `lines`.
		means_to_plot: plot the mean of all sims for each comp. in `means`.
		figname: name of figure to save, None if no need to save fig.
		figtitle: title of figure, None if using default title.
		show_plot: show plot if True.
		save_data: saves the data of the SIR lines of all simulation.
	"""

	compartments = ("S", "I", "R")
	colors = {"S": u'#1f77b4', "I": u'#ff7f0e', "R": u'#2ca02c'}

	if max_t is None:
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
	SIR_lines = {"S": S_lines, "I": I_lines, "R": R_lines}

	# Plot the averages of S, I, and R curves
	fig = plt.figure(figsize=(13, 8))
	for comp in compartments:
		if comp in means_to_plot:
			plt.plot(SIR_lines[comp].mean(0),
					 label=comp, color=colors[comp], linewidth=3)

	# Plot all I curves to visualize simulation runs
	for comp in compartments:
		if comp in lines_to_plot:
			for comp_line in SIR_lines[comp]:
				plt.plot(comp_line, color=colors[comp], linewidth=0.5)

	# Configure plot, show, and save
	plt.legend()
	plt.grid(which="major")
	if figtitle is None:
		plt.title(f"SIR Curves of {len(SIRs)} Simulations")
	else:
		plt.title(figtitle)
	if show_plot:
		plt.show()
	if figname is not None:
		fig.savefig(figname)

	# Save data
	if save_data:
		# Choose appropriate name, matching with figname if possible
		if figname is None:
			fname = "SIR_data.pkl"
		else:
			basename = figname.split(".")[0] or "SIR_data"
			fname = basename + ".pkl"
		# Pickle data
		with open(fname, "wb") as f:
			pickle.dump(SIR_lines, f)


def repeat_simulation(sim_config={},
					  num_sim=100,
					  parallel=None):
	"""
	Repeats an outbreak simulation given its config.

	Args:
		sim_config: config of the outbreak simulation.
		num_sim: number of simulations to run.
		parallel: number of processes to run in parallel (# of CPUs if 0).
				  Doesn't run in parallel if None.

	Return:
		list of SIR curves for all simulations.
	"""
	
	# Define simulation task based on given config.
	# This will return a function `sim(sim_id)` that takes an identifier.
	sim = functools.partial(outbreak_simulation, **sim_config)
	sim_ids = list(range(num_sim))

	# Run simulations in parallel
	if parallel is None or num_sim < 5:
		SIRs = [sim(sim_id) for sim_id in sim_ids]
	else:
		processes = parallel if parallel > 0 else None
		with Pool(processes=processes) as pool:
			SIRs = pool.map(sim, sim_ids)

	return SIRs


def main():
	init_random_seed(123)
	plot_averaged_SIRs(repeat_simulation(), figname=None, show_plot=True)


if __name__ == '__main__':
	main()

