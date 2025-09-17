#!/bin/bash

# =============================================================================
# Enhanced MARL Two-Tower Recommendation System - Training Script
# =============================================================================
# Description: Comprehensive training script for multi-agent reinforcement 
#              learning recommendation system with fairness optimization
# Author: Enhanced MARL Team
# Date: September 17, 2025
# Hardware: RTX 4060+ (8GB+ VRAM recommended)
# =============================================================================

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
EXPERIMENT_ID="marl_${TIMESTAMP}"

# Default configuration
DEFAULT_CONFIG="movielens.yaml"
DEFAULT_MODE="full"
DEFAULT_GPUS="0"
DEFAULT_EPOCHS=100
DEFAULT_BATCH_SIZE=256
DEFAULT_LEARNING_RATE=1e-4

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

print_banner() {
    echo -e "${CYAN}"
    echo "============================================================================="
    echo "🚀 Enhanced MARL Two-Tower Recommendation System - Training Pipeline"
    echo "============================================================================="
    echo -e "${NC}"
}

print_section() {
    echo -e "${BLUE}📊 $1${NC}"
    echo "-----------------------------------------------------------------------------"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${PURPLE}ℹ️  $1${NC}"
}

# =============================================================================
# HELP FUNCTION
# =============================================================================

show_help() {
    print_banner
    echo "USAGE:"
    echo "  $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -c, --config CONFIG_FILE     Configuration file (default: $DEFAULT_CONFIG)"
    echo "  -m, --mode MODE              Training mode: full|baseline|ablation (default: $DEFAULT_MODE)"
    echo "  -g, --gpus GPU_IDS           GPU IDs to use (default: $DEFAULT_GPUS)"
    echo "  -e, --epochs EPOCHS          Number of training epochs (default: $DEFAULT_EPOCHS)"
    echo "  -b, --batch-size BATCH_SIZE  Batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  -l, --lr LEARNING_RATE       Learning rate (default: $DEFAULT_LEARNING_RATE)"
    echo "  -r, --resume CHECKPOINT      Resume from checkpoint"
    echo "  -d, --debug                  Enable debug mode"
    echo "  -v, --validate-only          Run validation only"
    echo "  -a, --ablation-study         Run comprehensive ablation study"
    echo "  --mixed-precision            Enable mixed precision training"
    echo "  --profile                    Enable profiling"
    echo "  --wandb                      Enable Weights & Biases logging"
    echo "  --no-fair-sampling           Disable fair sampling"
    echo "  --no-buhs                    Disable BUHS module"
    echo "  --no-gini-agent              Disable GINI fairness agent"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "TRAINING MODES:"
    echo "  full      : Complete Enhanced MARL system with all components"
    echo "  baseline  : Two-Tower baseline without MARL enhancements"
    echo "  ablation  : Progressive ablation study across all components"
    echo ""
    echo "EXAMPLES:"
    echo "  # Full training with default settings"
    echo "  $0"
    echo ""
    echo "  # Training with specific configuration"
    echo "  $0 --config movielens.yaml --mode full --epochs 150"
    echo ""
    echo "  # Ablation study with mixed precision"
    echo "  $0 --mode ablation --mixed-precision --wandb"
    echo ""
    echo "  # Resume training from checkpoint"
    echo "  $0 --resume checkpoints/model_epoch_50.pt"
    echo ""
    echo "  # Multi-GPU training"
    echo "  $0 --gpus 0,1 --batch-size 512"
    echo ""
}

# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================

validate_environment() {
    print_section "Environment Validation"
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_info "Python version: $python_version"
    
    # Check if we're in the correct directory
    if [[ ! -f "$PROJECT_ROOT/trainer.py" ]]; then
        print_error "trainer.py not found. Please run from project root or scripts directory."
        exit 1
    fi
    
    # Check required files
    required_files=("trainer.py" "config.py" "genre_agent.py" "contextgnn.py" "metrics.py")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
            print_error "Required file not found: $file"
            exit 1
        fi
    done
    print_success "All required files found"
    
    # Check PyTorch and CUDA
    if python3 -c "import torch" &> /dev/null; then
        cuda_available=$(python3 -c "import torch; print(torch.cuda.is_available())")
        if [[ "$cuda_available" == "True" ]]; then
            gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())")
            gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
            print_success "CUDA available: $gpu_count GPU(s) - $gpu_name"
        else
            print_warning "CUDA not available - will use CPU training (slower)"
        fi
    else
        print_error "PyTorch not installed. Please install requirements first."
        exit 1
    fi
    
    # Check memory requirements
    available_memory=$(python3 -c "
import psutil
import torch
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU Memory: {gpu_mem:.1f}GB')
else:
    print('GPU Memory: Not available')
ram_mem = psutil.virtual_memory().total / 1024**3
print(f'RAM: {ram_mem:.1f}GB')
" 2>/dev/null || echo "Memory check failed")
    
    print_info "$available_memory"
    
    echo ""
}

# =============================================================================
# CONFIGURATION SETUP
# =============================================================================

setup_configuration() {
    print_section "Configuration Setup"
    
    # Create necessary directories
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/checkpoints"
    mkdir -p "$PROJECT_ROOT/results"
    mkdir -p "$PROJECT_ROOT/figures"
    
    # Check config file
    if [[ ! -f "$PROJECT_ROOT/$CONFIG_FILE" ]]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        print_info "Available configurations:"
        ls "$PROJECT_ROOT"/*.yaml 2>/dev/null || echo "No .yaml files found"
        exit 1
    fi
    
    print_success "Configuration file: $CONFIG_FILE"
    print_success "Training mode: $MODE"
    print_success "Experiment ID: $EXPERIMENT_ID"
    
    # Set up logging
    LOG_DIR="$PROJECT_ROOT/logs/$EXPERIMENT_ID"
    mkdir -p "$LOG_DIR"
    
    print_success "Log directory: $LOG_DIR"
    echo ""
}

# =============================================================================
# GPU SETUP
# =============================================================================

setup_gpu() {
    print_section "GPU Configuration"
    
    # Parse GPU IDs
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
    
    print_info "Requested GPUs: ${GPU_ARRAY[*]}"
    
    # Validate GPU IDs
    if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        available_gpus=$(python3 -c "import torch; print(torch.cuda.device_count())")
        print_info "Available GPUs: $available_gpus"
        
        for gpu_id in "${GPU_ARRAY[@]}"; do
            if [[ $gpu_id -ge $available_gpus ]]; then
                print_error "GPU $gpu_id not available (only $available_gpus GPUs detected)"
                exit 1
            fi
        done
        
        export CUDA_VISIBLE_DEVICES="$GPUS"
        print_success "CUDA_VISIBLE_DEVICES set to: $GPUS"
    else
        print_warning "CUDA not available - using CPU"
        unset CUDA_VISIBLE_DEVICES
    fi
    
    echo ""
}

# =============================================================================
# TRAINING EXECUTION
# =============================================================================

run_training() {
    print_section "Training Execution"
    
    # Build Python command
    cmd="python3 -u trainer.py"
    cmd="$cmd --config $CONFIG_FILE"
    cmd="$cmd --mode $MODE"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --lr $LEARNING_RATE"
    cmd="$cmd --experiment-id $EXPERIMENT_ID"
    cmd="$cmd --log-dir $LOG_DIR"
    
    # Add optional flags
    [[ -n "$RESUME_CHECKPOINT" ]] && cmd="$cmd --resume $RESUME_CHECKPOINT"
    [[ "$DEBUG" == "true" ]] && cmd="$cmd --debug"
    [[ "$VALIDATE_ONLY" == "true" ]] && cmd="$cmd --validate-only"
    [[ "$MIXED_PRECISION" == "true" ]] && cmd="$cmd --mixed-precision"
    [[ "$PROFILE" == "true" ]] && cmd="$cmd --profile"
    [[ "$WANDB" == "true" ]] && cmd="$cmd --wandb"
    [[ "$NO_FAIR_SAMPLING" == "true" ]] && cmd="$cmd --no-fair-sampling"
    [[ "$NO_BUHS" == "true" ]] && cmd="$cmd --no-buhs"
    [[ "$NO_GINI_AGENT" == "true" ]] && cmd="$cmd --no-gini-agent"
    
    print_info "Training command:"
    echo "$cmd"
    echo ""
    
    # Execute training
    cd "$PROJECT_ROOT"
    
    print_success "Starting training..."
    echo "================================================================================"
    
    # Run with proper logging
    if [[ "$DEBUG" == "true" ]]; then
        $cmd 2>&1 | tee "$LOG_DIR/training.log"
    else
        $cmd > "$LOG_DIR/training.log" 2>&1 &
        TRAIN_PID=$!
        
        print_info "Training started with PID: $TRAIN_PID"
        print_info "Monitor progress: tail -f $LOG_DIR/training.log"
        
        # Monitor training
        monitor_training $TRAIN_PID "$LOG_DIR/training.log"
    fi
}

# =============================================================================
# ABLATION STUDY EXECUTION
# =============================================================================

run_ablation_study() {
    print_section "Comprehensive Ablation Study"
    
    # Define ablation configurations
    declare -a ablation_configs=(
        "baseline:Two-Tower baseline without MARL"
        "contextgnn:Baseline + ContextGNN"
        "marl:ContextGNN + MARL Controller"
        "fair-sampling:MARL + Fair Sampling"
        "buhs:Fair Sampling + BUHS Module"
        "gini-agent:BUHS + GINI Agent"
        "full:Complete Enhanced MARL System"
    )
    
    print_info "Running ${#ablation_configs[@]} ablation configurations..."
    
    for config_entry in "${ablation_configs[@]}"; do
        IFS=':' read -ra config_parts <<< "$config_entry"
        config_name="${config_parts[0]}"
        config_desc="${config_parts[1]}"
        
        print_section "Ablation: $config_name - $config_desc"
        
        # Create specific experiment ID
        ablation_experiment_id="${EXPERIMENT_ID}_ablation_${config_name}"
        ablation_log_dir="$PROJECT_ROOT/logs/$ablation_experiment_id"
        mkdir -p "$ablation_log_dir"
        
        # Build command
        cmd="python3 -u trainer.py"
        cmd="$cmd --config $CONFIG_FILE"
        cmd="$cmd --mode $config_name"
        cmd="$cmd --epochs $EPOCHS"
        cmd="$cmd --batch-size $BATCH_SIZE"
        cmd="$cmd --lr $LEARNING_RATE"
        cmd="$cmd --experiment-id $ablation_experiment_id"
        cmd="$cmd --log-dir $ablation_log_dir"
        
        # Add ablation-specific flags
        case $config_name in
            "baseline")
                cmd="$cmd --no-fair-sampling --no-buhs --no-gini-agent --no-marl"
                ;;
            "contextgnn")
                cmd="$cmd --no-fair-sampling --no-buhs --no-gini-agent --no-marl --contextgnn-only"
                ;;
            "marl")
                cmd="$cmd --no-fair-sampling --no-buhs --no-gini-agent"
                ;;
            "fair-sampling")
                cmd="$cmd --no-buhs --no-gini-agent"
                ;;
            "buhs")
                cmd="$cmd --no-gini-agent"
                ;;
            "gini-agent")
                # All components except final optimizations
                ;;
            "full")
                # All components enabled
                [[ "$MIXED_PRECISION" == "true" ]] && cmd="$cmd --mixed-precision"
                ;;
        esac
        
        print_info "Command: $cmd"
        
        # Execute ablation
        cd "$PROJECT_ROOT"
        $cmd > "$ablation_log_dir/training.log" 2>&1
        
        if [[ $? -eq 0 ]]; then
            print_success "Ablation $config_name completed successfully"
        else
            print_error "Ablation $config_name failed"
        fi
        
        echo ""
    done
    
    print_success "Ablation study completed"
    print_info "Results available in: $PROJECT_ROOT/logs/"
}

# =============================================================================
# TRAINING MONITORING
# =============================================================================

monitor_training() {
    local pid=$1
    local log_file=$2
    
    print_info "Monitoring training progress..."
    
    # Monitor in background
    while kill -0 $pid 2>/dev/null; do
        sleep 30
        
        # Check for recent progress
        if [[ -f "$log_file" ]]; then
            # Show last few lines with metrics
            tail -n 5 "$log_file" | grep -E "(Epoch|Loss|HR@10|NDCG|GINI)" | tail -n 1
            
            # Check for errors
            if tail -n 20 "$log_file" | grep -i "error\|exception\|failed" >/dev/null; then
                print_warning "Potential error detected in training log"
                tail -n 5 "$log_file" | grep -i "error\|exception\|failed"
            fi
        fi
    done
    
    # Check final status
    wait $pid
    exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "Training completed successfully"
    else
        print_error "Training failed with exit code: $exit_code"
        print_info "Check log file: $log_file"
        exit $exit_code
    fi
}

# =============================================================================
# POST-TRAINING ANALYSIS
# =============================================================================

run_post_training_analysis() {
    print_section "Post-Training Analysis"
    
    # Check if results analysis notebook exists
    if [[ -f "$PROJECT_ROOT/results_analysis.ipynb" ]]; then
        print_info "Running results analysis notebook..."
        
        # Convert notebook to HTML report
        if command -v jupyter &> /dev/null; then
            cd "$PROJECT_ROOT"
            jupyter nbconvert --to html --execute results_analysis.ipynb \
                              --output "results/analysis_report_${EXPERIMENT_ID}.html" \
                              2>/dev/null || print_warning "Notebook execution failed"
            
            if [[ -f "results/analysis_report_${EXPERIMENT_ID}.html" ]]; then
                print_success "Analysis report generated: results/analysis_report_${EXPERIMENT_ID}.html"
            fi
        else
            print_warning "Jupyter not available - skipping notebook execution"
        fi
    fi
    
    # Generate summary report
    if [[ -f "$PROJECT_ROOT/results/final_summary_report.json" ]]; then
        print_success "Training summary available in: results/final_summary_report.json"
    fi
    
    echo ""
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    # Parse command line arguments
    CONFIG_FILE="$DEFAULT_CONFIG"
    MODE="$DEFAULT_MODE"
    GPUS="$DEFAULT_GPUS"
    EPOCHS="$DEFAULT_EPOCHS"
    BATCH_SIZE="$DEFAULT_BATCH_SIZE"
    LEARNING_RATE="$DEFAULT_LEARNING_RATE"
    RESUME_CHECKPOINT=""
    DEBUG="false"
    VALIDATE_ONLY="false"
    ABLATION_STUDY="false"
    MIXED_PRECISION="false"
    PROFILE="false"
    WANDB="false"
    NO_FAIR_SAMPLING="false"
    NO_BUHS="false"
    NO_GINI_AGENT="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -m|--mode)
                MODE="$2"
                shift 2
                ;;
            -g|--gpus)
                GPUS="$2"
                shift 2
                ;;
            -e|--epochs)
                EPOCHS="$2"
                shift 2
                ;;
            -b|--batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            -l|--lr)
                LEARNING_RATE="$2"
                shift 2
                ;;
            -r|--resume)
                RESUME_CHECKPOINT="$2"
                shift 2
                ;;
            -d|--debug)
                DEBUG="true"
                shift
                ;;
            -v|--validate-only)
                VALIDATE_ONLY="true"
                shift
                ;;
            -a|--ablation-study)
                ABLATION_STUDY="true"
                shift
                ;;
            --mixed-precision)
                MIXED_PRECISION="true"
                shift
                ;;
            --profile)
                PROFILE="true"
                shift
                ;;
            --wandb)
                WANDB="true"
                shift
                ;;
            --no-fair-sampling)
                NO_FAIR_SAMPLING="true"
                shift
                ;;
            --no-buhs)
                NO_BUHS="true"
                shift
                ;;
            --no-gini-agent)
                NO_GINI_AGENT="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Show banner
    print_banner
    
    # Validate environment
    validate_environment
    
    # Setup configuration
    setup_configuration
    
    # Setup GPU
    setup_gpu
    
    # Execute training based on mode
    if [[ "$ABLATION_STUDY" == "true" ]]; then
        run_ablation_study
    else
        run_training
    fi
    
    # Post-training analysis
    run_post_training_analysis
    
    print_section "Training Pipeline Complete"
    print_success "Experiment ID: $EXPERIMENT_ID"
    print_success "Logs: $LOG_DIR"
    print_success "Results: $PROJECT_ROOT/results/"
    
    echo ""
    print_info "🎉 Enhanced MARL Two-Tower training pipeline completed successfully!"
}

# Execute main function
main "$@"
