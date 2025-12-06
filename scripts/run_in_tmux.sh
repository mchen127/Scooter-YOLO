#!/bin/bash
#
# Tmux helper script for running TL-1 baseline experiments
#
# Usage:
#   bash scripts/run_in_tmux.sh              # Start all experiments
#   bash scripts/run_in_tmux.sh tl1_lr_1e4   # Start single experiment
#

set -e

# Configuration
SESSION_NAME="tl1_baseline"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

# Create logs directory
mkdir -p "$LOG_DIR"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed"
    echo "Install with: sudo apt-get install tmux"
    exit 1
fi

# Get experiment name or default to 'all'
EXPERIMENT="${1:-all}"

# Determine command based on experiment
if [[ "$EXPERIMENT" == "all" || "$EXPERIMENT" == "lr" ]]; then
    CMD="python scripts/train_tl1_baseline.py --ablation all"
    WINDOW_NAME="tl1_all"
else
    CMD="python scripts/train_tl1_baseline.py --experiment $EXPERIMENT"
    WINDOW_NAME="$EXPERIMENT"
fi

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/tl1_${EXPERIMENT}_${TIMESTAMP}.log"

echo "================================================================"
echo "Starting TL-1 Baseline Training in tmux"
echo "================================================================"
echo "Session name: $SESSION_NAME"
echo "Experiment:   $EXPERIMENT"
echo "Command:      $CMD"
echo "Log file:     $LOG_FILE"
echo "================================================================"
echo ""

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  Warning: tmux session '$SESSION_NAME' already exists"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session:  tmux attach -t $SESSION_NAME"
    echo "  2. Kill existing session:       tmux kill-session -t $SESSION_NAME"
    echo "  3. Use a different session name by editing this script"
    echo ""
    read -p "Kill existing session and continue? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION_NAME"
        echo "✓ Killed existing session"
    else
        echo "Aborted."
        exit 1
    fi
fi

# Create new tmux session and run command
tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_NAME" -c "$PROJECT_DIR"

# Activate conda environment and send command to tmux session
tmux send-keys -t "$SESSION_NAME:$WINDOW_NAME" "cd $PROJECT_DIR" C-m
tmux send-keys -t "$SESSION_NAME:$WINDOW_NAME" "conda activate dev3.12" C-m
tmux send-keys -t "$SESSION_NAME:$WINDOW_NAME" "$CMD 2>&1 | tee $LOG_FILE" C-m

echo "✓ Training started in tmux session"
echo ""
echo "================================================================"
echo "Tmux Session Commands:"
echo "================================================================"
echo "Attach to session:    tmux attach -t $SESSION_NAME"
echo "Detach from session:  Ctrl+b, then d"
echo "List sessions:        tmux ls"
echo "Kill session:         tmux kill-session -t $SESSION_NAME"
echo ""
echo "Monitor logs:         tail -f $LOG_FILE"
echo "================================================================"
echo ""
echo "The training will continue running even if you disconnect."
echo "Use 'tmux attach -t $SESSION_NAME' to view progress."
echo ""
