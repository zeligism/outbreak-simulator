
EXP="exp3"

ja_file="${EXP}.ja"
data_dir="$EXP"
rm -f "$ja_file"
mkdir -p "$data_dir"

INFECTION_RATES=($(seq 0.01 0.01 0.10))
TEST_DELAYS=(0 1 2 4 8)

# Construct base run of this job
baserun="time python run.py"
baserun+=" --random_seed 123"
baserun+=" --regenerate_graph"
baserun+=" --num_sim 100"
baserun+=" --gamma_infection"
baserun+=" --recovery_time 14"
baserun+=" --recovery_rate 0.3333"
baserun+=" --quarantine_length 14"
baserun+=" --testing_capacity 1.0"
baserun+=" --testing_rounds 10"
baserun+=" --testing_schedule 1 1 1 1 1 0 0"
baserun+=" --max_t 120"

for infection_rate in "${INFECTION_RATES[@]}"; do
  for test_delay in "${TEST_DELAYS[@]}"; do
    # Add varying parameters to base run
    command="$baserun"
    command+=" --infection_rate $infection_rate"
    command+=" --test_delay $test_delay"
    # Choose a unique name for this figure
    figname="SIRs_beta=${infection_rate}_delay=${test_delay}"
    figtitle="beta = ${infection_rate}, delay = ${test_delay}"
    command+=" --figname '${data_dir}/${figname}.png'"
    command+=" --figtitle '$figtitle'"
    # Add command to job array file
    echo "$command" >> "$ja_file"
  done
done

