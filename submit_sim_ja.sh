
source activate graph
# This uses a utility script provided by Dalma HPC
slurm_parallel_ja_submit.sh -t 00:30:00 "${1:-sim.ja}"
