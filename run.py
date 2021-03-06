
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
	parser.add_argument("-G", "--graph_type", type=str.lower, default="barabasi_albert",
		choices=("ba", "barabasi_albert", "er", "erdos_renyi"),
		help="Type of graph to generate")
	parser.add_argument("--graph_args", type=float, nargs="*", default=(),
		help="Parameters of the graph generator")
	parser.add_argument("-R", "--regenerate_graph", action="store_true",
		help="Regenerate graph every simulation")
	parser.add_argument("-N", "--population", type=int, default=4000,
		help="Population of community")

	parser.add_argument("--initial_infected", type=int, default=1,
		help="Number of individuals initially infected")
	parser.add_argument("-b", "--infection_rate", type=float, default=0.05,
		help="The daily probability of infecting a neighbor")
	parser.add_argument("--infection_curve", type=str.lower, default="gamma",
		choices=("constant", "gamma"),
		help="Type of infection curve (curve of infectiosness vs. duration since infected).")
	parser.add_argument("--recovery_time", type=int, default=14,
		help="Minimum time to recover")
	parser.add_argument("--recovery_rate", type=float, default=0.333,
		help="Daily probability of recovering after `recovery_time`")
	parser.add_argument("--testing_capacity", type=float, default=1.0,
		help="Size of testing pool as a percentage of whole population")
	parser.add_argument("--testing_rounds", type=int, default=10,
		help="Number of testing rounds to run until whole pool is tested")
	parser.add_argument("--testing_schedule", type=int, nargs="*", default=(1,),
		help="Cyclic testing schedule as a list of bits (e.g. 1 0 == bi-daily)")
	parser.add_argument("--sort_tests", type=int, nargs=2, metavar=("MAX_NEIGHBORS", "MAX_DEPTH"),
		help="Sort test pool using neighbor's priority strategy")
	parser.add_argument("--test_sensitivity", type=float, default=1.0,
		help="Sensitivity of test (i.e. rate of true positives among positives)")
	parser.add_argument("--test_specificity", type=float, default=1.0,
		help="Specificity of test (i.e. rate of true negatives among negatives)")
	parser.add_argument("--test_delay", type=int, default=0,
		help="Number of days needed to get test results back)")
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
	parser.add_argument("--means_to_plot", type=str, default="IR",
		help="Which compartment means to plot")
	parser.add_argument("--figname", type=str, default=None,
		help="Name of figure to save")
	parser.add_argument("--figtitle", type=str, default=None,
		help="Title of figure")
	parser.add_argument("--max_t", type=int, default=None,
		help="Limit of x axis (time)")
	parser.add_argument("--show_plot", action="store_true",
		help="Show plot")
	parser.add_argument("--save_data", action="store_true",
		help="Save data of all SIR values for all simulations")

	# Parse arguments
	args = parser.parse_args()

	return args


def main(args):

	# Initialize random seed if provided
	if args.random_seed is not None:
		init_random_seed(args.random_seed)

	# Initialize generator of community network
	if args.graph_type in ("ba", "barabasi_albert"):
		graph_args = (3,) if len(args.graph_args) == 0 else args.graph_args
		G = functools.partial(nx.barabasi_albert_graph, args.population, int(graph_args[0]))
	elif args.graph_type in ("er", "erdos_renyi"):
		# Use nx.erdos_renyi_graph if p is large (i.e. graph is not sparse)
		graph_args = (0.0025,) if len(args.graph_args) == 0 else args.graph_args
		G = functools.partial(nx.fast_gnp_random_graph, args.population, float(graph_args[0]))
	else:
		raise NotImplementedError(f"Graph type {args.graph_type} not implemented")

	# If no need to regenerate graph for each simulation, generate graph now
	if not args.regenerate_graph:
		G = G()

	# Initialize simulation configurations
	infection_rate = InfectionRate(args.infection_rate, args.infection_curve == "gamma")
	recovery_rate = RecoveryRate(args.recovery_time, args.recovery_rate)
	testing_schedule = tuple(map(bool, args.testing_schedule))
	sort_tests = args.sort_tests is not None
	sort_max_neighbors, sort_max_depth = args.sort_tests if sort_tests else (0, 0)
	sim_config = {
		"random_seed": args.random_seed,
		"G": G,
		"initial_infected": args.initial_infected,
		"infection_rate": infection_rate,
		"recovery_rate": recovery_rate,
		"testing_capacity": args.testing_capacity,
		"testing_rounds": args.testing_rounds,
		"testing_schedule": testing_schedule,
		"sort_tests": sort_tests,
		"sort_max_neighbors": sort_max_neighbors,
		"sort_max_depth": sort_max_depth,
		"test_sensitivity": args.test_sensitivity,
		"test_specificity": args.test_specificity,
		"test_delay": args.test_delay,
		"quarantine_length": args.quarantine_length,
		"report_interval": args.report_interval,
		"stop_if_positive": args.stop_if_positive,
	}

	SIRs = repeat_simulation(sim_config=sim_config,
							 num_sim=args.num_sim,
							 parallel=args.parallel,)

	plot_averaged_SIRs(SIRs,
					   max_t=args.max_t,
					   lines_to_plot=args.lines_to_plot,
					   means_to_plot=args.means_to_plot,
					   figname=args.figname,
					   figtitle=args.figtitle,
					   show_plot=args.show_plot,
					   save_data=args.save_data)


if __name__ == '__main__':
	args = parse_args()
	main(args)

