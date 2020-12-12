
ja_file="exp1.ja"
rm -f "$ja_file"
mkdir -p "exp1"

INFECTION_RATE=("0.025" "0.05" "0.075" "0.1")
TESTING_CAPACITY=("0.01" "0.1")
TESTING_ROUNDS=(1 10)
QUARANTINE_LENGTH=(1 14)

for infection_rate in "${INFECTION_RATE[@]}"; do
  for testing_capacity in "${TESTING_CAPACITY[@]}"; do
    for testing_rounds in "${TESTING_ROUNDS[@]}"; do
      for quarantine_length in "${QUARANTINE_LENGTH[@]}"; do
        # Choose a unique name for this figure
        figname="SIRs_"
        figname+="beta=${infection_rate},"
        figname+="testcap=${testing_capacity},"
        figname+="rounds=${testing_rounds},"
        figname+="qlen=${quarantine_length}"
        # Construct command of this job
        command="time python run.py"
        command+=" --random_seed 123"
        command+=" --infection_rate $infection_rate"
        command+=" --gamma_infection"
        command+=" --recovery_time 14"
        command+=" --recovery_rate 0.3333"
        command+=" --testing_capacity $testing_capacity"
        command+=" --testing_rounds $testing_rounds"
        command+=" --testing_schedule 1 1 1 1 1 0 0"
        command+=" --quarantine_length $quarantine_length"
        command+=" --num_sim 100"
        command+=" --figname exp1/${figname}.png"
        #command+=" --parallel"
        echo "$command" >> "$ja_file"
      done
    done
  done
done


for infection_rate in "${INFECTION_RATE[@]}"; do
  quarantine_length=0
  # Choose a unique name for this figure
  figname="SIRs_"
  figname+="beta=${infection_rate},"
  figname+="qlen=0"
  # Construct command of this job
  command="time python run.py"
  command+=" --random_seed 123"
  command+=" --infection_rate $infection_rate"
  command+=" --gamma_infection"
  command+=" --recovery_time 14"
  command+=" --recovery_rate 0.3333"
  command+=" --testing_schedule 1 1 1 1 1 0 0"
  command+=" --quarantine_length 0"
  command+=" --num_sim 100"
  command+=" --figname exp1/${figname}.png"
  #command+=" --parallel"
  echo "$command" >> "$ja_file"
done

