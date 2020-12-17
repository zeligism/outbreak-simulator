
EXP="exp1"

ja_file="${EXP}.ja"
data_dir="$EXP"
rm -f "$ja_file"
mkdir -p "$data_dir"

INFECTION_RATES=(0.015 0.030 0.060)
QUARANTINE_LENGTHS=(0 1 7 14)
TESTING_CAPACITIES=(0.1 0.5 1.0)
TESTING_ROUNDS=(1 5 10)

# Construct base run of this job
baserun="time python run.py"
baserun+=" --random_seed 123"
baserun+=" --regenerate_graph"
baserun+=" --num_sim 100"
baserun+=" --gamma_infection"
baserun+=" --recovery_time 14"
baserun+=" --recovery_rate 0.3333"
baserun+=" --testing_schedule 1 1 1 1 1 0 0"
baserun+=" --max_t 120"

for infection_rate in "${INFECTION_RATES[@]}"; do
  for quarantine_length in "${QUARANTINE_LENGTHS[@]}"; do
    if (( $quarantine_length == 0 )); then
      # Add varying parameters to base run
      command="$baserun"
      command+=" --infection_rate $infection_rate"
      command+=" --quarantine_length $quarantine_length"
      # Choose a unique name for this figure
      figname="SIRs_beta=${infection_rate}_qlen=${quarantine_length}"
      figtitle="beta=${infection_rate}, qlen=${quarantine_length}"
      command+=" --figname '${data_dir}/${figname}.png'"
      command+=" --figtitle '$figtitle'"
      # Add command to job array file
      echo "$command" >> "$ja_file"
    else
      for testing_capacity in "${TESTING_CAPACITIES[@]}"; do
        for testing_rounds in "${TESTING_ROUNDS[@]}"; do
          # Add varying parameters to base run
          command="$baserun"
          command+=" --infection_rate $infection_rate"
          command+=" --quarantine_length $quarantine_length"
          command+=" --testing_capacity $testing_capacity"
          command+=" --testing_rounds $testing_rounds"
          # Choose a unique name for this figure
          figname="SIRs_beta=${infection_rate}_qlen=${quarantine_length}_testcap=${testing_capacity}_rounds=${testing_rounds}"
          figtitle="beta=${infection_rate}, qlen=${quarantine_length}, testcap=${testing_capacity}, rounds=${testing_rounds}"
          command+=" --figname '${data_dir}/${figname}.png'"
          command+=" --figtitle '$figtitle'"
          # Add command to job array file
          echo "$command" >> "$ja_file"
        done
      done
    fi
  done
done

