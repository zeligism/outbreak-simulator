
ja_file="exp2.ja"
rm -f "$ja_file"

# Data directory of this experiment
DATA_DIR="exp2"
mkdir -p "$DATA_DIR"

INFECTION_RATES=("0.001" "0.002" "0.005" "0.010" "0.015" "0.025" "0.05" "0.1")
QUARANTINE_LENGTHS=(0 1 100)

# Construct base run of this job
baserun="time python run.py"
baserun+=" --random_seed 123"
baserun+=" --regenerate_graph"
baserun+=" --num_sim 100"
baserun+=" --gamma_infection"
baserun+=" --recovery_time 14"
baserun+=" --recovery_rate 0.3333"
baserun+=" --testing_schedule 1 1 1 1 1 0 0"

for infection_rate in "${INFECTION_RATES[@]}"; do
  for quarantine_length in "${QUARANTINE_LENGTHS[@]}"; do
    # Add varying parameters to base run
    command="$baserun"
    command+=" --infection_rate $infection_rate"
    command+=" --quarantine_length $quarantine_length"
    # Choose a unique name for this figure
    figname="SIRs_beta=${infection_rate}_qlen=${quarantine_length}"
    figtitle="beta = ${infection_rate}, qlen = ${quarantine_length}"
    command+=" --figname '${DATA_DIR}/${figname}.png'"
    command+=" --figtitle '$figtitle'"
    # Add command to job array file
    echo "$command" >> "$ja_file"
  done
done

