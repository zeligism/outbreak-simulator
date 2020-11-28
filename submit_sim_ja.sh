
source activate graph
mkdir -p plots
# This uses a utility script provided by Dalma HPC
slurm_parallel_ja_submit.sh -t 00:07:00 sim.ja
