#!/usr/bin/env bash
# Evaluate one checkpoint on all 30 held-out conditions, one shard per simulator GPU.
#
# Serial evaluation of 30 conditions takes over two hours (1050 steps at ~4 steps/s each),
# which is the difference between having a verdict the same evening and the next morning.
# One policy server feeds every simulator: inference is a small fraction of the wall clock
# next to physics and rendering, so the server is not the bottleneck and does not need
# replicating - it only needs a memory fraction small enough to leave room for a simulator
# on its own GPU.
#
#   scripts/eval/run_eval.sh <step> <server-gpu> <sim-gpu> [<sim-gpu> ...]
#
# Any number of simulator GPUs: three while training still holds 0-3, six once it has
# finished. Results land in <WORK>/eval_<step>/, merged by shard_conditions.py.
set -euo pipefail

STEP=${1:?usage: run_eval.sh <step> <server-gpu> <sim-gpu> [<sim-gpu> ...]}
SGPU=${2:?}
shift 2
GPUS=("$@")
[ ${#GPUS[@]} -gt 0 ] || { echo "need at least one simulator GPU"; exit 1; }
N=${#GPUS[@]}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
POLARIS=${POLARIS_ROOT:?set POLARIS_ROOT to your PolaRiS checkout}
OPENPI=$POLARIS/third_party/openpi
WORK=${WORK:?set WORK to the working directory}
CONDITIONS=${CONDITIONS:-$POLARIS/PolaRiS-Hub/pour_mustard/eval_conditions.json}
CKPT=$OPENPI/checkpoints/pi05_droid_jointpos_polaris_pourmustard/pourmustard_v1/$STEP
OUT=$WORK/eval_$STEP
SHARDS=$WORK/eval_shards_$N
PORT=$((8000 + STEP / 1000))

[ -d "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }
mkdir -p "$OUT"

PER=$(python3 "$HERE/shard_conditions.py" split "$CONDITIONS" "$SHARDS" --shards "$N" \
      | tee /dev/stderr | head -1 | grep -oE "[0-9]+ conditions" | grep -oE "[0-9]+")

# The config only registers if our package is importable, and openpi is run through uv, so
# PYTHONPATH is how it reaches the subprocess. Without it: "Config not found".
echo "serving $STEP on GPU $SGPU port $PORT"
cd "$OPENPI"
CUDA_VISIBLE_DEVICES=$SGPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 PYTHONPATH=$REPO/src \
    nohup setsid uv run scripts/serve_policy.py --port "$PORT" policy:checkpoint \
    --policy.config pi05_droid_jointpos_polaris_pourmustard --policy.dir "$CKPT" \
    > "$OUT/serve.log" 2>&1 &

until grep -q "server listening" "$OUT/serve.log" 2>/dev/null; do
    grep -qE "Traceback|Error" "$OUT/serve.log" && { tail -20 "$OUT/serve.log"; exit 1; }
    sleep 5
done
echo "server up; both norm-stats lines:"
grep "norm stats" "$OUT/serve.log"

# Staggered, because the first inference request makes the server JIT-compile, and that
# blocks its event loop long enough for a later client's websocket handshake to time out -
# which is how a shard died the first time this ran. Waiting for each shard to take its
# first step means the compile is over before the next one dials in.
cd "$POLARIS"
for ((i = 0; i < N; i++)); do
    gpu=${GPUS[$i]}
    CUDA_VISIBLE_DEVICES=$gpu nohup setsid "$REPO/scripts/polaris_env.sh" \
        python "$REPO/scripts/eval/eval_policy.py" \
        --policy.client DroidJointPos --policy.port "$PORT" --policy.open-loop-horizon 8 \
        --environment DROID-PourMustard \
        --initial-conditions-file "$SHARDS/shard$i.json" \
        --rollouts "$PER" --run-folder "$OUT/shard$i" \
        > "$OUT/shard$i.log" 2>&1 &
    echo "  shard $i on GPU $gpu"
    if [ "$i" -eq 0 ]; then
        until grep -q "it/s\]" "$OUT/shard$i.log" 2>/dev/null; do
            grep -qE "Traceback" "$OUT/shard$i.log" && { tail -5 "$OUT/shard$i.log"; exit 1; }
            sleep 10
        done
    else
        sleep 20
    fi
done
echo "launched $N shards; merge with:"
echo "  python scripts/eval/shard_conditions.py merge $OUT --shards $N --per $PER"
