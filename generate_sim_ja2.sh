
EXP="exp2"

ja_file="${EXP}.ja"
data_dir="$EXP"
rm -f "$ja_file"
mkdir -p "$data_dir"

INFECTION_RATES=($(seq 0.01 0.01 0.10))
QUARANTINE_LENGTHS=(0 1 7 100)

# Construct base run of this job
baserun="time python run.py"
baserun+=" --random_seed 123"
baserun+=" --regenerate_graph"
baserun+=" --num_sim 100"
baserun+=" --gamma_infection"
baserun+=" --recovery_time 14"
baserun+=" --recovery_rate 0.3333"
baserun+=" --testing_capacity 1.0"
baserun+=" --testing_rounds 10"
baserun+=" --testing_schedule 1 1 1 1 1 0 0"
baserun+=" --max_t 120"

for infection_rate in "${INFECTION_RATES[@]}"; do
  for quarantine_length in "${QUARANTINE_LENGTHS[@]}"; do
    # Add varying parameters to base run
    command="$baserun"
    command+=" --infection_rate $infection_rate"
    command+=" --quarantine_length $quarantine_length"
    # Choose a unique name for this figure
    figname="SIRs_beta=${infection_rate}_qlen=${quarantine_length}"
    figtitle="beta = ${infection_rate}, qlen = ${quarantine_length}"
    command+=" --figname '${data_dir}/${figname}.png'"
    command+=" --figtitle '$figtitle'"
    # Add command to job array file
    echo "$command" >> "$ja_file"
  done
done

