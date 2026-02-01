#!/bin/bash
#
# Flow Matching Architecture Study - Quick Run Script
#
# Usage:
#   ./study/run_study.sh              # Run all experiments in tmux
#   ./study/run_study.sh flow         # Run flow ablation only
#   ./study/run_study.sh --list       # List all experiments
#   ./study/run_study.sh --attach     # Attach to running session
#   ./study/run_study.sh --status     # Show checkpoint status
#
# Environment:
#   GPU=0 ./study/run_study.sh        # Use specific GPU
#

set -e

# Configuration
SESSION_NAME="flow-study"
GPU="${GPU:-1}"
LOG_DIR="study/logs"
RESULTS_DIR="study/results"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directories
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

show_help() {
    echo "Flow Matching Architecture Study"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  all             Run all experiments (default)"
    echo "  flow            Run flow method ablation"
    echo "  arch            Run architecture ablation"
    echo "  dataset         Run dataset scaling ablation"
    echo "  augmentation    Run augmentation ablation"
    echo "  scale           Run model scale ablation"
    echo ""
    echo "Options:"
    echo "  --list          List all experiments and status"
    echo "  --attach        Attach to running tmux session"
    echo "  --status        Show checkpoint completion status"
    echo "  --kill          Kill running tmux session"
    echo "  --dry-run       Print commands without executing"
    echo ""
    echo "Environment:"
    echo "  GPU=N           GPU to use (default: 1)"
    echo ""
    echo "Examples:"
    echo "  $0                      # Run all in tmux"
    echo "  $0 flow                 # Run flow ablation"
    echo "  GPU=0 $0 arch           # Run arch ablation on GPU 0"
    echo "  $0 --list               # Show experiment status"
}

show_status() {
    echo -e "${BLUE}Checkpoint Status:${NC}"
    echo ""

    local total=0
    local complete=0

    for ckpt_dir in study/checkpoints/*/; do
        if [ -d "$ckpt_dir" ]; then
            name=$(basename "$ckpt_dir")
            total=$((total + 1))
            if [ -f "${ckpt_dir}best.pt" ]; then
                echo -e "  ${GREEN}✓${NC} $name"
                complete=$((complete + 1))
            else
                echo -e "  ${YELLOW}○${NC} $name (incomplete)"
            fi
        fi
    done

    echo ""
    echo -e "${BLUE}Progress: ${complete}/${total} complete${NC}"
}

attach_session() {
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux attach-session -t "$SESSION_NAME"
    else
        echo -e "${RED}No running session: $SESSION_NAME${NC}"
        echo "Start with: $0 [ablation]"
        exit 1
    fi
}

kill_session() {
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux kill-session -t "$SESSION_NAME"
        echo -e "${GREEN}Killed session: $SESSION_NAME${NC}"
    else
        echo -e "${YELLOW}No session to kill${NC}"
    fi
}

run_in_tmux() {
    local ablation="$1"
    local dry_run="$2"
    local log_file="${LOG_DIR}/study_${ablation}_$(date +%Y%m%d_%H%M%S).log"

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo -e "${YELLOW}Session already running: $SESSION_NAME${NC}"
        echo "Use '$0 --attach' to attach or '$0 --kill' to stop"
        exit 1
    fi

    local cmd="cd $(pwd) && "
    cmd+="CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=offline "
    cmd+="uv run python -m study.run_all_experiments "

    if [ "$ablation" = "all" ]; then
        cmd+="--all "
    else
        cmd+="--ablation $ablation "
    fi

    cmd+="--gpu $GPU "

    if [ "$dry_run" = "true" ]; then
        cmd+="--dry-run "
    fi

    cmd+="2>&1 | tee $log_file; exec bash"

    echo -e "${BLUE}Starting experiments in tmux session: $SESSION_NAME${NC}"
    echo -e "  Ablation: ${GREEN}$ablation${NC}"
    echo -e "  GPU: ${GREEN}$GPU${NC}"
    echo -e "  Log: ${GREEN}$log_file${NC}"
    echo ""
    echo -e "Attach with: ${YELLOW}tmux attach -t $SESSION_NAME${NC}"
    echo -e "Or run: ${YELLOW}$0 --attach${NC}"
    echo ""

    tmux new-session -d -s "$SESSION_NAME" "$cmd"

    echo -e "${GREEN}Session started!${NC}"
}

# Parse arguments
ABLATION="all"
DRY_RUN="false"

case "$1" in
    --help|-h)
        show_help
        exit 0
        ;;
    --list)
        uv run python -m study.run_all_experiments --list
        exit 0
        ;;
    --status)
        show_status
        exit 0
        ;;
    --attach)
        attach_session
        exit 0
        ;;
    --kill)
        kill_session
        exit 0
        ;;
    --dry-run)
        DRY_RUN="true"
        ABLATION="${2:-all}"
        ;;
    flow|arch|dataset|augmentation|scale|all)
        ABLATION="$1"
        if [ "$2" = "--dry-run" ]; then
            DRY_RUN="true"
        fi
        ;;
    "")
        ABLATION="all"
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac

# Run
run_in_tmux "$ABLATION" "$DRY_RUN"
