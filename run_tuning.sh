#!/bin/bash

# Define a list of epsilon values to iterate through
EPSILONS=(6 5 4 3 2 1)

# Loop through each epsilon value
for EPS in "${EPSILONS[@]}"
do
    echo "Starting tuning for epsilon: ${EPS}"
    python scripts/tune_ddpm.py wilt 3096 synthetic catboost wilt_dp_eps_${EPS} --eval_seeds --dp_eps ${EPS}
    echo "Tuning for epsilon ${EPS} finished."
done

echo "All tuning processes have been completed."
