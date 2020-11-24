
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from outbreak_simulator import *

TEST_DIR = "test"

def simulate_transition_time(rates, num_people=10000):
	transition_times = np.zeros(num_people) + np.nan
	t = 1
	num_rem = num_people
	while num_rem > 0 and t < len(rates):
		num_transitioned = np.random.binomial(num_rem, rates[t])
		rem_range = range(num_people - num_rem,
						  num_people - num_rem + num_transitioned)
		transition_times[rem_range] = t
		num_rem -= num_transitioned
		t += 1
	return transition_times


def test_infection_rate(rate=0.1, max_t=40):
	print("=== Testing infection rate ===")
	const_infection_rate = InfectionRate(infection_rate=rate)
	gamma_infection_rate = InfectionRate(infection_rate=rate,
										 use_gamma_rate=True)

	const_rates = [const_infection_rate(t) for t in range(max_t+1)]
	gamma_rates = [gamma_infection_rate(t) for t in range(max_t+1)]

	check_const = all(const_rates[0] == r for r in const_rates)
	print("Constant infection rate is constant:", check_const)

	max_gamma = max(gamma_rates)
	max_gamma_t = gamma_rates.index(max_gamma) + 1
	print("Gamma infection rate has peak at day:", max_gamma_t)

	gamma_mass_lt_14 = sum(gamma_rates[:15]) * (0.118 / rate)
	print("Gamma infection rate's mass <= 14 days:", gamma_mass_lt_14)

	fig = plt.figure()
	plt.plot(const_rates, label="Constant")
	plt.plot(gamma_rates, label="Gamma")
	plt.title("Infection rates")
	plt.legend()
	fig.savefig(os.path.join(TEST_DIR, "infection_rates.png"))

	const_infection_times = simulate_transition_time(const_rates)
	gamma_infection_times = simulate_transition_time(gamma_rates)

	fig = plt.figure()
	bins = int(max(const_infection_times) - min(const_infection_times))
	plt.hist(const_infection_times, bins=bins, rwidth=0.7, label="Constant")
	bins = int(max(gamma_infection_times) - min(gamma_infection_times))
	plt.hist(gamma_infection_times, bins=bins, rwidth=0.5, label="Gamma")
	plt.title("Simulated infection times for 10,000 people")
	plt.xlabel("Infection time")
	plt.ylabel("Number of infected")
	plt.legend()
	fig.savefig(os.path.join(TEST_DIR, "infection_times.png"))


def test_recovery_rate(recovery_time=14, max_t=40):
	print("=== Testing recovery rate ===")

	normal_recovery_rate = RecoveryRate(recovery_time=recovery_time)
	decay_recovery_rate  = RecoveryRate(recovery_time=recovery_time,
										recovery_rate=1/3)

	normal_rates = [normal_recovery_rate(t) for t in range(max_t+1)]
	decay_rates = [decay_recovery_rate(t) for t in range(max_t+1)]

	fig = plt.figure()
	plt.plot(normal_rates, label="Normal")
	plt.plot(decay_rates, label="Decay")
	plt.title("Recovery rates")
	plt.legend()
	fig.savefig(os.path.join(TEST_DIR, "recovery_rates.png"))

	normal_recovery_times = simulate_transition_time(normal_rates)
	decay_recovery_times = simulate_transition_time(decay_rates)

	all_recovered_eq_14 = all(t == recovery_time for t in normal_recovery_times)
	print(f"All normal recovery times are exactly {recovery_time} days:", all_recovered_eq_14)
	all_recovered_gt_14 = all(t >= recovery_time for t in decay_recovery_times)
	print(f"All decay recovery times are >= {recovery_time} days:", all_recovered_gt_14)

	fig = plt.figure()
	bins = int(max(decay_recovery_times) - min(decay_recovery_times))
	plt.hist(normal_recovery_times, bins=bins, label="Normal")
	plt.hist(decay_recovery_times, bins=bins, rwidth=0.5, label="Decay")
	plt.title("Simulated recovery times for 10,000 people")
	plt.xlabel("Recovery time")
	plt.ylabel("Number of recovered")
	plt.legend()
	fig.savefig(os.path.join(TEST_DIR, "recovery_times.png"))


def test_testing_pool(pool_size=17):
	nodes = list(range(pool_size))

	print("=== Testing pool with 1 round and daily schedule ===")
	testing_pool = TestingPool(nodes, rounds=1, schedule=[True])
	round1 = testing_pool.next_round()
	round2 = testing_pool.next_round()
	print("Round 1 equals to round 2:", set(round1) == set(round2))
	print("Each round exhausts the testing pool:", set(round1) == set(nodes))

	print("=== Testing pool with 3 rounds and daily schedule ===")
	testing_pool = TestingPool(nodes, rounds=3, schedule=[True])
	round1 = testing_pool.next_round()
	round2 = testing_pool.next_round()
	round3 = testing_pool.next_round()
	round4 = testing_pool.next_round()
	rounds123 = set(round1) | set(round2) | set(round3)
	print("Round 1 equals to round 4:", set(round1) == set(round4))
	print("Rounds 1, 2, and 3 exhaust the testing pool:", set(rounds123) == set(nodes))

	print("=== Testing pool with 3 rounds and working-day schedule ===")
	testing_pool = TestingPool(nodes, rounds=3, schedule=[True]*5+[False]*2)
	round1 = testing_pool.next_round()
	round2 = testing_pool.next_round()
	round3 = testing_pool.next_round()
	round4 = testing_pool.next_round()
	round5 = testing_pool.next_round()
	round6 = testing_pool.next_round()
	round7 = testing_pool.next_round()
	round8 = testing_pool.next_round()
	rounds123 = set(round1) | set(round2) | set(round3)
	rounds458 = set(round4) | set(round5) | set(round8)
	print("Round 1 equals to round 4:", set(round1) == set(round4))
	print("Round 2 equals to round 5:", set(round2) == set(round5))
	print("Round 3 equals to round 8:", set(round3) == set(round8))
	print("Rounds 6 and 7 are empty:", len(set(round6) | set(round7)) == 0)
	print("Rounds 1, 2, and 3 exhaust the testing pool:", set(rounds123) == set(nodes))
	print("Rounds 4, 5, and 8 exhaust the testing pool:", set(rounds458) == set(nodes))
	print("Rounds 1, 2, and 3 equal to rounds 4, 5, and 8:", set(rounds123) == set(rounds458))


def test_threshold(G=nx.barabasi_albert_graph(4000, 3), MFL=True, QMF=True):

	# Define simulation function
	def simulate(epi_threshold, num_sim=10, gamma=1/14):
		print("Epidemic threshold =", epi_threshold)
		# Choose gamma s.t. mean recovery time is 14 days
		# and choose beta s.t. beta / gamma == epidemic threshold
		beta = gamma * epi_threshold
		sim_config = {
			"infection_rate": InfectionRate(beta),
			"recovery_rate": RecoveryRate(0, gamma),
			"quarantine_length": 0,
		}
		# Run simulations
		SIRs = repeat_simulation(G=G, sim_config=sim_config, num_sim=num_sim)
		figname = f"SIRs(beta={beta:.4f},gamma={gamma:.4f}).png"
		figname = os.path.join(TEST_DIR, figname)
		plot_averaged_SIRs(SIRs, means_to_plot="IR", figname=figname, show_plot=False)

	if MFL:
		print("=== Mean-Field Like Threshold ===")
		# Calculate mean field like (MFL) epidemic threshold
		k = sum(d for _, d in G.degree) / len(G.degree)  # first moment
		k_2 = sum(d*d for _, d in G.degree) / len(G.degree)  # second moment
		mfl_threshold = k / (k_2 - k)

		print("Simulating with beta == gamma * threshold")
		simulate(mfl_threshold)
		print("Simulating with beta == 2 * gamma * threshold")
		simulate(mfl_threshold*2)
		print("Simulating with beta == 3 * gamma * threshold")
		simulate(mfl_threshold*3)

	if QMF:
		print("=== Quenched Mean-Field Threshold ===")
		# Calculate quenched mean field (QMF) epidemic threshold
		L = nx.normalized_laplacian_matrix(G)
		eigvals = np.linalg.eigvals(L.A)  # from networkx docs
		qmf_threshold = 1 / max(map(abs, eigvals))
		qmf_threshold = np.real(qmf_threshold)  # to avoid complex fractions

		print("Simulating with beta == gamma * threshold")
		simulate(qmf_threshold)
		print("Simulating with beta == 2 * gamma * threshold")
		simulate(qmf_threshold*2)


def test_repeat_simulation():
	print("=== Running 100 simulations in parallel ===")
	figname = os.path.join(TEST_DIR, "SIRs.png")
	plot_averaged_SIRs(repeat_simulation(parallel=0, num_sim=200),
					   figname=figname, show_plot=False)
	print("Done.")


def main():
	random.seed(123)
	np.random.seed(123)
	if not os.path.isdir(TEST_DIR): os.mkdir(TEST_DIR)
	test_infection_rate()
	test_recovery_rate()
	test_testing_pool()
	test_threshold(MFL=True, QMF=False)
	test_repeat_simulation()


if __name__ == '__main__':
	main()

