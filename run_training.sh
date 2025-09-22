#!/bin/bash

# Define a list of epsilon values to iterate through
EPSILONS=(0.6 0.7 0.8 0.9 1 1.5 2 3 4 7 9 10)

# Loop through each epsilon value
for EPS in "${EPSILONS[@]}"
do
    echo "Starting training for epsilon: ${EPS}"
    python scripts/pipeline.py --config exp/wilt/wilt_dp_eps_${EPS}_best/config.toml --train_dp_eps ${EPS} --sample --eval
    echo "Training for epsilon ${EPS} finished."
done

echo "All training processes have been completed."