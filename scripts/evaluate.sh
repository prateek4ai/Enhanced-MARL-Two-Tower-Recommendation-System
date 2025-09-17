#!/bin/bash

# =============================================================================
# Enhanced MARL Two-Tower Recommendation System - Evaluation Script
# =============================================================================
# Description: Comprehensive evaluation script for multi-agent reinforcement 
#              learning recommendation system with fairness analysis
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
EVAL_ID="eval_${TIMESTAMP}"

# Default configuration
DEFAULT_CONFIG="movielens.yaml"
DEFAULT_MODE="comprehensive"
DEFAULT_CHECKPOINT=""
DEFAULT_GPUS="0"
DEFAULT_BATCH_SIZE=512
DEFAULT_SPLIT="test"

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
    echo "📊 Enhanced MARL Two-Tower Recommendation System - Evaluation Pipeline"
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
    echo "  -p, --checkpoint CHECKPOINT  Model checkpoint path (required)"
    echo "  -m, --mode MODE              Evaluation mode (default: $DEFAULT_MODE)"
    echo "  -g, --gpus GPU_IDS           GPU IDs to use (default: $DEFAULT_GPUS)"
    echo "  -b, --batch-size BATCH_SIZE  Batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  -s, --split DATA_SPLIT       Data split to evaluate (default: $DEFAULT_SPLIT)"
    echo "  -o, --output OUTPUT_DIR      Output directory for results"
    echo "  -f, --format FORMAT          Output format: json|csv|both (default: both)"
    echo "  --baseline-compare           Compare with baseline models"
    echo "  --fairness-analysis          Run detailed fairness analysis"
    echo "  --agent-breakdown            Analyze individual agent performance"
    echo "  --statistical-test           Run statistical significance tests"
    echo "  --generate-report            Generate comprehensive evaluation report"
    echo "  --profile                    Enable performance profiling"
    echo "  --debug                      Enable debug mode"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "EVALUATION MODES:"
    echo "  comprehensive : Complete evaluation with all metrics and analyses"
    echo "  quick        : Basic metrics evaluation (HR@10, NDCG@10, GINI)"
    echo "  fairness     : Focus on fairness and long-tail metrics"
    echo "  agents       : Multi-agent performance analysis"
    echo "  inference    : Inference speed and latency testing"
    echo "  custom       : User-defined metric subset"
    echo ""
    echo "EXAMPLES:"
    echo "  # Comprehensive evaluation"
    echo "  $0 --checkpoint checkpoints/model_best.pt"
    echo ""
    echo "  # Quick evaluation with specific config"
    echo "  $0 --checkpoint checkpoints/model_epoch_100.pt --mode quick --config movielens.yaml"
    echo ""
    echo "  # Fairness analysis with report generation"
    echo "  $0 --checkpoint checkpoints/model_best.pt --mode fairness --generate-report"
    echo ""
    echo "  # Multi-agent analysis with baseline comparison"
    echo "  $0 --checkpoint checkpoints/model_best.pt --mode agents --baseline-compare"
    echo ""
    echo "  # Inference performance testing"
    echo "  $0 --checkpoint checkpoints/model_best.pt --mode inference --profile"
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
            print_warning "CUDA not available - will use CPU evaluation (slower)"
        fi
    else
        print_error "PyTorch not installed. Please install requirements first."
        exit 1
    fi
    
    echo ""
}

# =============================================================================
# CHECKPOINT VALIDATION
# =============================================================================

validate_checkpoint() {
    print_section "Checkpoint Validation"
    
    if [[ ! -f "$CHECKPOINT_PATH" ]]; then
        print_error "Checkpoint file not found: $CHECKPOINT_PATH"
        exit 1
    fi
    
    # Check checkpoint compatibility
    checkpoint_info=$(python3 -c "
import torch
try:
    checkpoint = torch.load('$CHECKPOINT_PATH', map_location='cpu')
    print(f\"Model: {checkpoint.get('model_name', 'Unknown')}\")
    print(f\"Epoch: {checkpoint.get('epoch', 'Unknown')}\")
    print(f\"Performance: HR@10={checkpoint.get('hr10', 'N/A'):.3f}, GINI={checkpoint.get('gini', 'N/A'):.3f}\")
    print('Valid checkpoint')
except Exception as e:
    print(f'Invalid checkpoint: {e}')
" 2>/dev/null || echo "Checkpoint validation failed")
    
    if echo "$checkpoint_info" | grep -q "Invalid checkpoint"; then
        print_error "Checkpoint is invalid or corrupted"
        exit 1
    fi
    
    print_success "Checkpoint validated successfully"
    echo "$checkpoint_info" | while IFS= read -r line; do
        [[ "$line" != "Valid checkpoint" ]] && print_info "$line"
    done
    
    echo ""
}

# =============================================================================
# EVALUATION EXECUTION
# =============================================================================

run_evaluation() {
    print_section "Evaluation Execution"
    
    # Create output directories
    EVAL_DIR="$OUTPUT_DIR/$EVAL_ID"
    mkdir -p "$EVAL_DIR/logs"
    mkdir -p "$EVAL_DIR/results"
    mkdir -p "$EVAL_DIR/figures"
    
    # Build evaluation command
    cmd="python3 -u trainer.py"
    cmd="$cmd --config $CONFIG_FILE"
    cmd="$cmd --mode evaluate"
    cmd="$cmd --checkpoint $CHECKPOINT_PATH"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --data-split $DATA_SPLIT"
    cmd="$cmd --eval-mode $EVAL_MODE"
    cmd="$cmd --output-dir $EVAL_DIR"
    cmd="$cmd --output-format $OUTPUT_FORMAT"
    
    # Add optional flags
    [[ "$BASELINE_COMPARE" == "true" ]] && cmd="$cmd --baseline-compare"
    [[ "$FAIRNESS_ANALYSIS" == "true" ]] && cmd="$cmd --fairness-analysis"
    [[ "$AGENT_BREAKDOWN" == "true" ]] && cmd="$cmd --agent-breakdown"
    [[ "$STATISTICAL_TEST" == "true" ]] && cmd="$cmd --statistical-test"
    [[ "$PROFILE" == "true" ]] && cmd="$cmd --profile"
    [[ "$DEBUG" == "true" ]] && cmd="$cmd --debug"
    
    print_info "Evaluation command:"
    echo "$cmd"
    echo ""
    
    # Set GPU environment
    if [[ -n "$GPUS" ]]; then
        export CUDA_VISIBLE_DEVICES="$GPUS"
        print_info "Using GPUs: $GPUS"
    fi
    
    # Execute evaluation
    cd "$PROJECT_ROOT"
    
    print_success "Starting evaluation..."
    echo "================================================================================"
    
    LOG_FILE="$EVAL_DIR/logs/evaluation.log"
    
    if [[ "$DEBUG" == "true" ]]; then
        $cmd 2>&1 | tee "$LOG_FILE"
        eval_exit_code=${PIPESTATUS[0]}
    else
        $cmd > "$LOG_FILE" 2>&1
        eval_exit_code=$?
    fi
    
    if [[ $eval_exit_code -eq 0 ]]; then
        print_success "Evaluation completed successfully"
    else
        print_error "Evaluation failed with exit code: $eval_exit_code"
        print_info "Check log file: $LOG_FILE"
        exit $eval_exit_code
    fi
    
    echo ""
}

# =============================================================================
# RESULTS PROCESSING
# =============================================================================

process_results() {
    print_section "Results Processing"
    
    # Check for results files
    if [[ -f "$EVAL_DIR/results/evaluation_metrics.json" ]]; then
        print_success "Evaluation metrics generated"
        
        # Display key metrics
        key_metrics=$(python3 -c "
import json
with open('$EVAL_DIR/results/evaluation_metrics.json', 'r') as f:
    metrics = json.load(f)
    
print(f'📊 Key Performance Metrics:')
print(f'   • HR@10: {metrics.get(\"hr10\", \"N/A\")}')
print(f'   • NDCG@10: {metrics.get(\"ndcg10\", \"N/A\")}')
print(f'   • GINI Coefficient: {metrics.get(\"gini\", \"N/A\")}')
print(f'   • Coverage: {metrics.get(\"coverage\", \"N/A\")}')
print(f'   • Tail HR@10: {metrics.get(\"tail_hr10\", \"N/A\")}')
if 'inference_latency_ms' in metrics:
    print(f'   • Inference Latency: {metrics[\"inference_latency_ms\"]}ms')
" 2>/dev/null || print_warning "Could not parse metrics")
    fi
    
    # Check for fairness analysis
    if [[ -f "$EVAL_DIR/results/fairness_analysis.json" ]]; then
        print_success "Fairness analysis completed"
    fi
    
    # Check for agent breakdown
    if [[ -f "$EVAL_DIR/results/agent_performance.json" ]]; then
        print_success "Agent performance breakdown generated"
    fi
    
    # List generated files
    print_info "Generated files:"
    find "$EVAL_DIR" -name "*.json" -o -name "*.csv" -o -name "*.png" | while read -r file; do
        echo "   • $(basename "$file")"
    done
    
    echo ""
}

# =============================================================================
# REPORT GENERATION
# =============================================================================

generate_evaluation_report() {
    if [[ "$GENERATE_REPORT" != "true" ]]; then
        return
    fi
    
    print_section "Evaluation Report Generation"
    
    # Generate comprehensive report
    report_cmd="python3 -c \"
import json
import os
from datetime import datetime

# Load results
results_dir = '$EVAL_DIR/results'
report_path = '$EVAL_DIR/evaluation_report.html'

print('Generating comprehensive evaluation report...')

# Create HTML report
html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced MARL Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #f0f8ff; padding: 20px; border-radius: 10px; }
        .metric { background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { color: #28a745; }
        .warning { color: #ffc107; }
        .error { color: #dc3545; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class='header'>
        <h1>🚀 Enhanced MARL Two-Tower Evaluation Report</h1>
        <p><strong>Generated:</strong> $(date)</p>
        <p><strong>Checkpoint:</strong> $CHECKPOINT_PATH</p>
        <p><strong>Mode:</strong> $EVAL_MODE</p>
    </div>
'''

# Add metrics if available
if os.path.exists(f'{results_dir}/evaluation_metrics.json'):
    with open(f'{results_dir}/evaluation_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    html_content += '''
    <h2>📊 Performance Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
    '''
    
    for metric, value in metrics.items():
        status = 'success' if isinstance(value, (int, float)) and value > 0.5 else 'warning'
        html_content += f'<tr><td>{metric}</td><td>{value}</td><td class=\"{status}\">✅</td></tr>'
    
    html_content += '</table>'

html_content += '''
    <h2>📁 Generated Files</h2>
    <ul>
'''

for root, dirs, files in os.walk(results_dir):
    for file in files:
        html_content += f'<li>{file}</li>'

html_content += '''
    </ul>
    <footer>
        <p><em>Report generated by Enhanced MARL Two-Tower Evaluation Pipeline</em></p>
    </footer>
</body>
</html>
'''

with open(report_path, 'w') as f:
    f.write(html_content)

print(f'✅ Report generated: {report_path}')
\""
    
    eval "$report_cmd" 2>/dev/null || print_warning "Report generation failed"
    
    echo ""
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    # Parse command line arguments
    CONFIG_FILE="$DEFAULT_CONFIG"
    CHECKPOINT_PATH="$DEFAULT_CHECKPOINT"
    EVAL_MODE="$DEFAULT_MODE"
    GPUS="$DEFAULT_GPUS"
    BATCH_SIZE="$DEFAULT_BATCH_SIZE"
    DATA_SPLIT="$DEFAULT_SPLIT"
    OUTPUT_DIR="$PROJECT_ROOT/evaluation_results"
    OUTPUT_FORMAT="both"
    BASELINE_COMPARE="false"
    FAIRNESS_ANALYSIS="false"
    AGENT_BREAKDOWN="false"
    STATISTICAL_TEST="false"
    GENERATE_REPORT="false"
    PROFILE="false"
    DEBUG="false"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -p|--checkpoint)
                CHECKPOINT_PATH="$2"
                shift 2
                ;;
            -m|--mode)
                EVAL_MODE="$2"
                shift 2
                ;;
            -g|--gpus)
                GPUS="$2"
                shift 2
                ;;
            -b|--batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            -s|--split)
                DATA_SPLIT="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -f|--format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            --baseline-compare)
                BASELINE_COMPARE="true"
                shift
                ;;
            --fairness-analysis)
                FAIRNESS_ANALYSIS="true"
                shift
                ;;
            --agent-breakdown)
                AGENT_BREAKDOWN="true"
                shift
                ;;
            --statistical-test)
                STATISTICAL_TEST="true"
                shift
                ;;
            --generate-report)
                GENERATE_REPORT="true"
                shift
                ;;
            --profile)
                PROFILE="true"
                shift
                ;;
            --debug)
                DEBUG="true"
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
    
    # Validate required arguments
    if [[ -z "$CHECKPOINT_PATH" ]]; then
        print_error "Checkpoint path is required. Use -p or --checkpoint"
        show_help
        exit 1
    fi
    
    # Show banner
    print_banner
    
    # Validate environment
    validate_environment
    
    # Validate checkpoint
    validate_checkpoint
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Run evaluation
    run_evaluation
    
    # Process results
    process_results
    
    # Generate report if requested
    generate_evaluation_report
    
    print_section "Evaluation Pipeline Complete"
    print_success "Evaluation ID: $EVAL_ID"
    print_success "Results: $EVAL_DIR"
    print_success "Logs: $EVAL_DIR/logs/evaluation.log"
    
    if [[ "$GENERATE_REPORT" == "true" ]] && [[ -f "$EVAL_DIR/evaluation_report.html" ]]; then
        print_success "Report: $EVAL_DIR/evaluation_report.html"
    fi
    
    echo ""
    print_info "🎉 Enhanced MARL Two-Tower evaluation completed successfully!"
}

# Execute main function
main "$@"
