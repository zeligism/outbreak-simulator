
ja_file="exp4.ja"
rm -f "$ja_file"

# Data directory of this experiment
DATA_DIR="exp4"
mkdir -p "$DATA_DIR"

INFECTION_RATES=("0.0005" "0.001" "0.002" "0.003" "0.005" "0.010" "0.015" "0.025" "0.050" "0.100")
GRAPH_TYPES=("barbasi_albert" "erdos_renyi")

# Construct base run of this job
baserun="time python run.py"
baserun+=" --random_seed 123"
baserun+=" --regenerate_graph"
baserun+=" --num_sim 100"
baserun+=" --gamma_infection"
baserun+=" --recovery_time 14"
baserun+=" --recovery_rate 0.3333"
baserun+=" --testing_schedule 1 1 1 1 1 0 0"
baserun+=" --quarantine_length 14"

for infection_rate in "${INFECTION_RATES[@]}"; do
  for graph_type in "${GRAPH_TYPES[@]}"; do
    # Add varying parameters to base run
    command="$baserun"
    command+=" --infection_rate $infection_rate"
    command+=" --graph_type $graph_type"
    # Choose a unique name for this figure
    figname="SIRs_beta=${infection_rate}_G=${graph_type}"
    figtitle="beta = ${infection_rate}, G = ${graph_type}"
    command+=" --figname '${DATA_DIR}/${figname}.png'"
    command+=" --figtitle '$figtitle'"
    # Add command to job array file
    echo "$command" >> "$ja_file"
  done
done

