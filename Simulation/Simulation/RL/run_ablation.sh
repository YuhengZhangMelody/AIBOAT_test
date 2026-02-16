#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_ablation.sh
# Optional env overrides:
#   TIMESTEPS=600000 N_ENVS=4 EPISODES=100 SEEDS="42 43 44" bash run_ablation.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTEPS="${TIMESTEPS:-600000}"
N_ENVS="${N_ENVS:-4}"
EPISODES="${EPISODES:-100}"
SEEDS_STR="${SEEDS:-42 43 44}"
read -r -a SEED_ARR <<< "$SEEDS_STR"

RUN_ROOT="$SCRIPT_DIR/ablation_runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"

SUMMARY_CSV="$RUN_ROOT/summary.csv"
echo "variant,seed,target_side,target_radius,transition_radius,success_rate,collision_rate,oob_rate,timeout_rate,mean_return,mean_action_delta,mean_radius_error,mean_tangent_alignment" > "$SUMMARY_CSV"

# Backup existing training artifacts once
if [[ -d "$SCRIPT_DIR/models" || -d "$SCRIPT_DIR/logs" ]]; then
  BACKUP_DIR="$RUN_ROOT/_preexisting_backup"
  mkdir -p "$BACKUP_DIR"
  [[ -d "$SCRIPT_DIR/models" ]] && mv "$SCRIPT_DIR/models" "$BACKUP_DIR/models"
  [[ -d "$SCRIPT_DIR/logs" ]] && mv "$SCRIPT_DIR/logs" "$BACKUP_DIR/logs"
fi

# Variant format: "name target_side target_radius transition_radius"
VARIANTS=(
  "B0 1 3.0 3.0"
  "B4 1 3.0 6.0"
  "B4_CW -1 3.0 6.0"
)

for variant in "${VARIANTS[@]}"; do
  read -r VAR_NAME TARGET_SIDE TARGET_RADIUS TRANSITION_RADIUS <<< "$variant"

  for SEED in "${SEED_ARR[@]}"; do
    RUN_ID="${VAR_NAME}_seed${SEED}"
    RUN_DIR="$RUN_ROOT/$RUN_ID"
    mkdir -p "$RUN_DIR"

    echo "============================================================"
    echo "Running $RUN_ID"
    echo "TIMESTEPS=$TIMESTEPS N_ENVS=$N_ENVS EPISODES=$EPISODES"
    echo "target_side=$TARGET_SIDE target_radius=$TARGET_RADIUS transition_radius=$TRANSITION_RADIUS"
    echo "============================================================"

    rm -rf "$SCRIPT_DIR/models" "$SCRIPT_DIR/logs"

    python3 "$SCRIPT_DIR/train_rl.py" \
      --timesteps "$TIMESTEPS" \
      --n-envs "$N_ENVS" \
      --seed "$SEED" \
      --target-side "$TARGET_SIDE" \
      --target-radius "$TARGET_RADIUS" \
      --transition-radius "$TRANSITION_RADIUS" | tee "$RUN_DIR/train.log"

    python3 "$SCRIPT_DIR/evaluate_rl.py" \
      --episodes "$EPISODES" \
      --seed "$SEED" \
      --target-side "$TARGET_SIDE" \
      --target-radius "$TARGET_RADIUS" \
      --transition-radius "$TRANSITION_RADIUS" | tee "$RUN_DIR/eval.log"

    [[ -d "$SCRIPT_DIR/models" ]] && cp -R "$SCRIPT_DIR/models" "$RUN_DIR/models"
    [[ -d "$SCRIPT_DIR/logs" ]] && cp -R "$SCRIPT_DIR/logs" "$RUN_DIR/logs"

    SUCCESS_RATE=$(grep -E "^Success rate:" "$RUN_DIR/eval.log" | awk '{print $3}')
    COLLISION_RATE=$(grep -E "^Collision rate:" "$RUN_DIR/eval.log" | awk '{print $3}')
    OOB_RATE=$(grep -E "^Out-of-bounds rate:" "$RUN_DIR/eval.log" | awk '{print $3}')
    TIMEOUT_RATE=$(grep -E "^Timeout rate:" "$RUN_DIR/eval.log" | awk '{print $3}')
    MEAN_RETURN=$(grep -E "^Mean return:" "$RUN_DIR/eval.log" | awk '{print $3}')
    MEAN_ACTION_DELTA=$(grep -E "^Mean action delta:" "$RUN_DIR/eval.log" | awk '{print $4}')
    MEAN_RADIUS_ERROR=$(grep -E "^Mean radius error:" "$RUN_DIR/eval.log" | awk '{print $4}')
    MEAN_TANGENT_ALIGNMENT=$(grep -E "^Mean tangent alignment:" "$RUN_DIR/eval.log" | awk '{print $4}')

    echo "$VAR_NAME,$SEED,$TARGET_SIDE,$TARGET_RADIUS,$TRANSITION_RADIUS,$SUCCESS_RATE,$COLLISION_RATE,$OOB_RATE,$TIMEOUT_RATE,$MEAN_RETURN,$MEAN_ACTION_DELTA,$MEAN_RADIUS_ERROR,$MEAN_TANGENT_ALIGNMENT" >> "$SUMMARY_CSV"
  done
done

echo "Done. Results saved in: $RUN_ROOT"
echo "Summary CSV: $SUMMARY_CSV"
