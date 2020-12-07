
import argparse
import random
import functools
import numpy as np
import networkx as nx
from outbreak_simulator import *


def parse_args():

	# Initialize parser and add arguments
	parser = argparse.ArgumentParser(
		description="Run simulations of an outbreak in a small community.")

	parser.add_argument("-s", "--random_seed", type=int, default=None,
		help="Random seed")
	parser.add_argument("-G", "--graph_type", type=str, default="barabasi_albert",
		help="Type of graph to generate")
	parser.add_argument("--regenerate_graph", action="store_true",
		help="Regenerate graph every simulation")
	parser.add_argument("-N", "--population", type=int, default=4000,
		help="Population of community")

	parser.add_argument("--initial_infected", type=int, default=1,
		help="Number of individuals initially infected")
	parser.add_argument("-b", "--infection_rate", type=float, default=0.05,
		help="The daily probability of infecting a neighbor")
	parser.add_argument("--gamma_infection", action="store_true",
		help="Use an infection rate that follows a gamma distribution-like curve")
	parser.add_argument("--recovery_time", type=int, default=14,
		help="Minimum time to recover")
	parser.add_argument("--recovery_rate", type=float, default=0.33,
		help="Daily probability of recovering after `recovery_time`")
	parser.add_argument("--testing_capacity", type=float, default=0.1,
		help="Size of testing pool as a percentage of whole population")
	parser.add_argument("--testing_rounds", type=int, default=10,
		help="Number of testing rounds to run until whole pool is tested")
	parser.add_argument("--testing_schedule", type=int, nargs="*", default=(1,),
		help="Cyclic testing schedule as a list of bits (e.g. 1 0 == bi-daily)")
	parser.add_argument("--test_sensitivity", type=float, default=1.0,
		help="Sensitivity of test (i.e. rate of true positives among positives)")
	parser.add_argument("--test_specificity", type=float, default=1.0,
		help="Specificity of test (i.e. rate of true negatives among negatives)")
	parser.add_argument("--quarantine_length", type=int, default=14,
		help="Time to spend in quarantine after testing positive")
	parser.add_argument("--report_interval", type=int, default=1000,
		help="Report SIR values every `report_interval` days")
	parser.add_argument("--stop_if_positive", action="store_true",
		help="Stop simulation as soon as a positive test is detected")

	parser.add_argument("-n", "--num_sim", type=int, default=10,
		help="Number of simulations run")
	parser.add_argument("--parallel", type=int, nargs="?", default=None, const=0,
		help="Number of processes to run in parallel (# of CPUs if no args given)")

	parser.add_argument("--lines_to_plot", type=str, default="IR",
		help="Which compartment lines to plot")
	parser.add_argument("--means_to_plot", type=str, default="SIR",
		help="Which compartment means to plot")
	parser.add_argument("--figname", type=str, default=None,
		help="Name of figure to save")
	parser.add_argument("--figtitle", type=str, default=None,
		help="Title of figure")
	parser.add_argument("--show_plot", action="store_true",
		help="Show plot")
	parser.add_argument("--save_data", action="store_true",
		help="Save data of all SIR values for all simulations")

	# Parse arguments
	args = parser.parse_args()

	return args


def main(args):

	# The random seed will be set here first so that graph generation can be
	# replicated. Though we have to take care about the randomness within the
	# simulation as well, so we have to pass the random seed to the simulations
	# too. In case we are using one graph, we have to pass different but unique
	# random seeds to each simulation so that they can be compared to other
	# runs of the experiments using various initial parameters. Otherwise,
	# in case when regenrate the graph for each simulation, it suffices to use
	# the same random seed for every experiment.

	# Initialize random seed if provided
	if args.random_seed is not None:
		random.seed(args.random_seed)
		np.random.seed(args.random_seed)

	# Initialize generator of community network
	if args.graph_type == "barabasi_albert":
		G = functools.partial(nx.barabasi_albert_graph, args.population, 3)
	elif args.graph_type == "erdos_renyi":
		#G = functools.partial(nx.erdos_renyi_graph, args.population, 0.0025)
		G = functools.partial(nx.fast_gnp_random_graph, args.population, 0.0025)
	else:
		NotImplementedError(f"Graph type {args.graph_type} not implemented")

	# If no need to regenerate graph for each simulation, generate graph now
	if not args.regenerate_graph:
		G = G()

	# Initialize simulation configurations
	infection_rate = InfectionRate(args.infection_rate, args.gamma_infection)
	recovery_rate = RecoveryRate(args.recovery_time, args.recovery_rate)
	testing_schedule = tuple(map(bool, args.testing_schedule))
	sim_config = {
		"random_seed": args.random_seed,
		"G": G,
		"initial_infected": args.initial_infected,
		"infection_rate": infection_rate,
		"recovery_rate": recovery_rate,
		"testing_capacity": args.testing_capacity,
		"testing_rounds": args.testing_rounds,
		"testing_schedule": testing_schedule,
		"test_sensitivity": args.test_sensitivity,
		"test_specificity": args.test_specificity,
		"quarantine_length": args.quarantine_length,
		"report_interval": args.report_interval,
		"stop_if_positive": args.stop_if_positive,
	}

	SIRs = repeat_simulation(sim_config=sim_config,
							 num_sim=args.num_sim,
							 parallel=args.parallel,)

	plot_averaged_SIRs(SIRs,
					   lines_to_plot=args.lines_to_plot,
					   means_to_plot=args.means_to_plot,
					   figname=args.figname,
					   figtitle=args.figtitle,
					   show_plot=args.show_plot,
					   save_data=args.save_data)


if __name__ == '__main__':
	args = parse_args()
	main(args)

