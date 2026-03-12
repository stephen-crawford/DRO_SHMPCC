#!/usr/bin/env bash
# run_paper_experiments.sh — Run exactly the experiments referenced by the paper
# and generate all publication figures.
#
# Paper: "Scenario-based Safe Horizon MPCC with OT-Improved Mode Coverage
#         for Targets with Switching Dynamics"
#
# Paper figures → C++ experiments mapping:
#   fig25 (§V.A Scenario budget scaling)    → Exp T  (missed-mode vs S)
#   fig29 (§I   + §V.B Coverage baselines)  → Exp X  (baselines on rare-mode stress)
#   fig30 (§V.C Geometry ablation)           → Exp Y  (shuffled/broken cost matrices)
#   fig27 (§V.D Rare-mode sweep)            → Exp V  (rare-mode probability sweep)
#   fig28 (§V.E Mode scaling)               → Exp W  (scaling with M modes)
#   fig26 (§V.F Ground-cost family)         → Exp U  (ground-cost ablation)
#   fig32 (§V.G Robustness across envs)     → Exp AA (per-seed boxplots)
#   fig33 (§V.H Pareto frontier)            → Exp AB (OT regularization sweep)
#   fig31 (§V.I Qualitative rollouts)       → Exp Z  (trajectory visualization)
#
# Usage:
#   ./scripts/run_paper_experiments.sh          # Run all paper experiments + figures
#   ./scripts/run_paper_experiments.sh --fast   # Reduced rollout counts for quick validation
#   ./scripts/run_paper_experiments.sh --only T V W  # Run specific experiments only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
RUNNER="$BUILD_DIR/paper_experiment_runner"
FIG_SCRIPT="$SCRIPT_DIR/generate_results_figures.py"

# Paper experiments in recommended execution order (fast → slow)
PAPER_EXPERIMENTS=(Z T V W X AA U Y AB)

# Parse arguments
FAST_MODE=false
ONLY_EXPS=()
SKIP_FIGURES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast)
            FAST_MODE=true
            shift
            ;;
        --only)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                ONLY_EXPS+=("$1")
                shift
            done
            ;;
        --skip-figures)
            SKIP_FIGURES=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--fast] [--only EXP1 EXP2 ...] [--skip-figures]"
            exit 1
            ;;
    esac
done

# If --only specified, use those; otherwise use all paper experiments
if [[ ${#ONLY_EXPS[@]} -gt 0 ]]; then
    EXPERIMENTS=("${ONLY_EXPS[@]}")
else
    EXPERIMENTS=("${PAPER_EXPERIMENTS[@]}")
fi

echo "============================================"
echo "  OT-SHMPCC Paper Experiment Runner"
echo "============================================"
echo "  Experiments: ${EXPERIMENTS[*]}"
echo "  Fast mode:   $FAST_MODE"
echo "  Project:     $PROJECT_DIR"
echo ""

# Step 1: Build
echo "[1/3] Building..."
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -1
cmake --build "$BUILD_DIR" --parallel 2>&1 | tail -3
echo ""

if [[ ! -x "$RUNNER" ]]; then
    echo "ERROR: $RUNNER not found after build"
    exit 1
fi

# Step 2: Create output directory
mkdir -p "$PROJECT_DIR/paper_figures"

# Step 3: Run experiments
echo "[2/3] Running experiments..."
TOTAL=${#EXPERIMENTS[@]}
START_TIME=$(date +%s)

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    idx=$((i + 1))
    echo ""
    echo "--- [$idx/$TOTAL] Experiment $exp ---"
    exp_start=$(date +%s)

    (cd "$PROJECT_DIR" && "$RUNNER" "$exp")

    exp_end=$(date +%s)
    elapsed=$((exp_end - exp_start))
    echo "  Experiment $exp completed in ${elapsed}s"
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "All experiments completed in ${TOTAL_ELAPSED}s"

# Step 4: Generate figures
if [[ "$SKIP_FIGURES" == false ]]; then
    echo ""
    echo "[3/3] Generating figures..."
    if command -v python3 &>/dev/null; then
        (cd "$PROJECT_DIR" && python3 "$FIG_SCRIPT")
        echo ""
        echo "Figures written to $PROJECT_DIR/paper_figures/"
        echo "Paper figure files:"
        ls -la "$PROJECT_DIR/paper_figures/"fig{25,26,27,28,29,30,31,32,33}_*.png 2>/dev/null || echo "  (some figures may not have been generated)"
    else
        echo "WARNING: python3 not found, skipping figure generation"
        echo "Run manually: python3 $FIG_SCRIPT"
    fi
else
    echo ""
    echo "[3/3] Skipping figure generation (--skip-figures)"
fi

echo ""
echo "============================================"
echo "  Done. CSV data in paper_figures/"
echo "============================================"
