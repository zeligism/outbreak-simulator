
import argparse
import random
import numpy as np
import networkx as nx
from outbreak_simulator import *


def parse_args():

	# Initialize parser and add arguments
	parser = argparse.ArgumentParser(
		description="Run simulations of an outbreak in a small community.")

	parser.add_argument("--random_seed", type=int, default=None,
		help="Random seed")
	parser.add_argument("--graph_type", type=str, default="barabasi_albert",
		help="Type of graph to generate")
	parser.add_argument("--population", type=int, default=4000,
		help="Population of community")

	parser.add_argument("--initial_infected", type=int, default=1,
		help="Number of individuals initially infected")
	parser.add_argument("--infection_rate", type=float, default=0.05,
		help="The daily probability of infecting a neighbor")
	parser.add_argument("--use_gamma_rate", action="store_true",
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

	parser.add_argument("--num_sim", type=int, default=10,
		help="Number of simulations run")
	parser.add_argument("--lines_to_plot", type=str, default="IR",
		help="Which compartment lines to plot")
	parser.add_argument("--means_to_plot", type=str, default="SIR",
		help="Which compartment means to plot")
	parser.add_argument("--figname", type=str, default=None,
		help="Name of figure to save")
	parser.add_argument("--processes", type=int, default=None,
		help="Number of processes to run")
	parser.add_argument("--show_plot", action="store_true",
		help="Whether to show plot or not")

	# Parse arguments
	args = parser.parse_args()

	return args


def main(args):

	# Initialize random seed if provided
	if args.random_seed is not None:
		random.seed(args.random_seed)
		np.random.seed(args.random_seed)

	# Initialize graph of community
	if args.graph_type == "barabasi_albert":
		G = nx.barabasi_albert_graph(args.population, 3)
	else:
		NotImplementedError(f"Graph type {args.graph_type} not implemented")

	# Initialize simulation configurations
	infection_rate = InfectionRate(args.infection_rate, args.use_gamma_rate)
	recovery_rate = RecoveryRate(args.recovery_time, args.recovery_rate)
	testing_schedule = tuple(map(bool, args.testing_schedule))
	sim_config = {
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

	SIRs = repeat_simulation(G=G,
							 sim_config=sim_config,
							 num_sim=args.num_sim,
							 processes=args.processes,)

	plot_averaged_SIRs(SIRs,
					   lines_to_plot=args.lines_to_plot,
					   means_to_plot=args.means_to_plot,
					   figname=args.figname,
					   show_plot=args.show_plot)


if __name__ == '__main__':
	args = parse_args()
	main(args)

