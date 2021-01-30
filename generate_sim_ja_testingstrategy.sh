
EXP="exp-testingstrategy"

ja_file="${EXP}.ja"
data_dir="$EXP"
rm -f "$ja_file"
mkdir -p "$data_dir"

GRAPH_TYPES=("BA" "ER")
BA_ARGS=(2 3 4)
ER_ARGS=(0.0010 0.0025 0.0040)
MAX_NEIGHBORS=(2 3 4)
MAX_DEPTH=(2 3 4)

# Construct base run of this job
baserun="time python run.py"
baserun+=" --random_seed 123"
#baserun+=" --regenerate_graph"
baserun+=" --num_sim 100"
baserun+=" --infection_rate 0.04"
baserun+=" --infection_curve gamma"
baserun+=" --recovery_time 14"
baserun+=" --recovery_rate 0.3333"
baserun+=" --testing_capacity 1.0"
baserun+=" --testing_rounds 10"
baserun+=" --testing_schedule 1 1 1 1 1 0 0"
baserun+=" --max_t 225"

for graph_type in "${GRAPH_TYPES[@]}"; do
  if [[ "$graph_type" == "BA" ]]; then
    GRAPH_ARGS=("${BA_ARGS[@]}")
  else
    GRAPH_ARGS=("${ER_ARGS[@]}")
  fi
  for graph_args in "${GRAPH_ARGS[@]}"; do
    for max_neighbors in "${MAX_NEIGHBORS[@]}"; do
      for max_depth in "${MAX_DEPTH[@]}"; do
        # Add varying parameters to base run
        command="$baserun"
        command+=" --graph_type $graph_type"
        command+=" --graph_args $graph_args"
        command+=" --sort_tests $max_neighbors $max_depth"
        # Choose a unique name for this figure
        figname="G=${graph_type}(${graph_args})_sort(${max_neighbors}, ${max_depth})"
        figtitle="${graph_type}(${graph_args}), sort(${max_neighbors}, ${max_depth})"
        command+=" --figname '${data_dir}/${figname}.png'"
        command+=" --figtitle '$figtitle'"
        # Add command to job array file
        echo "$command" >> "$ja_file"
      done
    done
  done
done
