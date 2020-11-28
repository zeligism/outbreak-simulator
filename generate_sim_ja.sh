
ja_file="sim.ja"
rm -f "$ja_file"

for infection_rate in $(seq 0.025 0.025 0.1); do
  for quarantine_length in $(seq 0 1 28); do
    command="time python run.py"
    command+=" --random_seed 123"
    command+=" --infection_rate $infection_rate"
    command+=" --use_gamma_rate"
    command+=" --recovery_time 14"
    command+=" --recovery_rate 0.3333"
    command+=" --testing_capacity 0.1"
    command+=" --testing_rounds 10"
    command+=" --testing_schedule 1 1 1 1 1 0 0"
    command+=" --quarantine_length $quarantine_length"
    command+=" --num_sim 100"
    command+=" --figname plots/SIRs(b=${infection_rate}, qlen=${quarantine_length}).png"
    #command+=" --parallel"
    echo "$command" >> "$ja_file"
  done
done

