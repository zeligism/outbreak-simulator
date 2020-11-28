
ja_file="sim.ja"
rm -f "$ja_file"

for infection_rate in $(seq 0.025 0.025 0.1); do
  for quarantine_length in $(seq 0 2 14); do
    command="time python run.py"
    command+=" --infection_rate $infection_rate"
    command+=" --use_gamma_rate"
    command+=" --recovery_time 14"
    command+=" --recovery_rate 0.333"
    command+=" --testing_capacity 0.1"
    command+=" --testing_rounds 10"
    command+=" --testing_schedule 1 1 1 1 1 0 0"
    command+=" --quarantine_length $quarantine_length"
    command+=" --num_sim 100"
    command+=" --figname SIRs_b${infection_rate}_q${quarantine_length}.png"
    #command+=" --parallel"
    echo "$command" >> "$ja_file"
  done
done

