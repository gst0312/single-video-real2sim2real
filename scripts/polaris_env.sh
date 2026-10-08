#!/bin/bash
# Run anything against our environment with the settings the GSWorld assets need.
#
#   scripts/polaris_env.sh python scripts/real2sim/check_environment.py --condition 0 --steps 60
#
# What it sets, and why:
#   PYTHONPATH        the gsplat-3dgs-renderer working copy first, so its `polaris` package
#                     shadows the installed one for this process: the scene splat is true
#                     3DGS and the stock surfel renderer would flatten it.
#   POLARIS_RENDERER  that branch's own switch, gsplat backend.
#   POLARIS_SPLAT_BG  black. Every splat here was trained with `_white_background = False`,
#                     while PolaRiS composites over 0.5 grey, which veils anything with
#                     alpha < 1 - the curtain most of all.
#   POLARIS_SPLAT_SOFTEN=0  skip the upstream half-resolution round trip, which is a blur.
#   POLARIS_DATA_PATH the asset hub; otherwise polaris resolves ./PolaRiS-Hub from the cwd.
#   CUDA_VISIBLE_DEVICES  the emptiest GPU, unless it is already set.
set -e
POLARIS=${POLARIS_ROOT:?set POLARIS_ROOT to your PolaRiS checkout}
POLARIS_3DGS=${POLARIS_3DGS:?set POLARIS_3DGS to the gsplat-3dgs-renderer checkout}
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}

export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used \
        --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
fi
export PATH=$POLARIS/.venv/bin:$PATH          # gsplat JIT-compiles and needs ninja from here
export PYTHONPATH=$POLARIS_3DGS/src:$REPO/src${PYTHONPATH:+:$PYTHONPATH}
export POLARIS_DATA_PATH=${POLARIS_DATA_PATH:-$POLARIS/PolaRiS-Hub}
export POLARIS_RENDERER=${POLARIS_RENDERER:-gsplat}
export POLARIS_SPLAT_BG=${POLARIS_SPLAT_BG:-0,0,0}
export POLARIS_SPLAT_SOFTEN=${POLARIS_SPLAT_SOFTEN:-0}
export PYTHONUNBUFFERED=1                     # simulation_app.close() drops buffered stdout

echo "GPU $CUDA_VISIBLE_DEVICES, hub $POLARIS_DATA_PATH, renderer $POLARIS_RENDERER" >&2
exec "$@"
