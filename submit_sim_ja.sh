
conda activate graph
mkdir -p plots
# This uses a utility script provided by Dalma HPC
slurm_parallel_ja_submit.sh -t 00:10:00 sim.ja
