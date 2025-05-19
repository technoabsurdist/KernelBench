#!/bin/bash
set -e

# Change to KernelBench root directory if script is run from scripts directory
if [[ $(basename $(pwd)) == "scripts" ]]; then
  cd ..
  echo "Changed to KernelBench root directory: $(pwd)"
fi

# Configuration
RUN_NAME="o4mini_benchmark"
MODEL_NAME="o4-mini"
SERVER_TYPE="openai"  # Adjust based on how o4-mini is accessed
TEMPERATURE=0
NUM_WORKERS=8  # Adjust based on your setup
LEVELS=(1 2)
LOG_DIR="runs/${RUN_NAME}/logs"
TIMEOUT=300
CHECKPOINT_FILE="runs/${RUN_NAME}/checkpoint.txt"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "runs/${RUN_NAME}/analysis"

# Initialize checkpoint file if it doesn't exist
if [ ! -f "$CHECKPOINT_FILE" ]; then
  echo "level_1_generated=false" > "$CHECKPOINT_FILE"
  echo "level_1_evaluated=false" >> "$CHECKPOINT_FILE"
  echo "level_2_generated=false" >> "$CHECKPOINT_FILE"
  echo "level_2_evaluated=false" >> "$CHECKPOINT_FILE"
  echo "analysis_completed=false" >> "$CHECKPOINT_FILE"
fi

# Load checkpoint state
source "$CHECKPOINT_FILE"

log_message() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_DIR}/main.log"
}

update_checkpoint() {
  local key=$1
  local value=$2
  sed -i "s/${key}=.*/${key}=${value}/" "$CHECKPOINT_FILE"
  log_message "Checkpoint updated: ${key}=${value}"
}

run_generation() {
  local level=$1
  local checkpoint_key="level_${level}_generated"
  local checkpoint_value=$(grep "${checkpoint_key}" "$CHECKPOINT_FILE" | cut -d'=' -f2)
  
  if [ "$checkpoint_value" = "false" ]; then
    log_message "Starting generation for level ${level}..."
    
    python3 scripts/generate_samples.py \
      run_name=${RUN_NAME} \
      dataset_src=huggingface \
      level=${level} \
      num_workers=${NUM_WORKERS} \
      server_type=${SERVER_TYPE} \
      model_name=${MODEL_NAME} \
      temperature=${TEMPERATURE} \
      2>&1 | tee "${LOG_DIR}/generation_level_${level}.log"
    
    # Only update checkpoint if command was successful
    if [ $? -eq 0 ]; then
      update_checkpoint "${checkpoint_key}" "true"
      log_message "Generation for level ${level} completed"
    else
      log_message "ERROR: Generation for level ${level} failed"
      exit 1
    fi
  else
    log_message "Skipping generation for level ${level} (already completed)"
  fi
}

run_evaluation() {
  local level=$1
  local checkpoint_key="level_${level}_evaluated"
  local checkpoint_value=$(grep "${checkpoint_key}" "$CHECKPOINT_FILE" | cut -d'=' -f2)
  
  if [ "$checkpoint_value" = "false" ]; then
    log_message "Starting evaluation for level ${level}..."
    
    python3 scripts/eval_from_generations.py \
      run_name=${RUN_NAME} \
      dataset_src=local \
      level=${level} \
      num_gpu_devices=1 \
      timeout=${TIMEOUT} \
      build_cache=True \
      num_cpu_workers=8 \
      2>&1 | tee "${LOG_DIR}/evaluation_level_${level}.log"
    
    # Only update checkpoint if command was successful
    if [ $? -eq 0 ]; then
      update_checkpoint "${checkpoint_key}" "true"
      log_message "Evaluation for level ${level} completed"
    else
      log_message "ERROR: Evaluation for level ${level} failed"
      exit 1
    fi
  else
    log_message "Skipping evaluation for level ${level} (already completed)"
  fi
}

run_analysis() {
  local checkpoint_key="analysis_completed"
  local checkpoint_value=$(grep "${checkpoint_key}" "$CHECKPOINT_FILE" | cut -d'=' -f2)
  
  if [ "$checkpoint_value" = "false" ]; then
    log_message "Starting analysis..."
    
    for level in "${LEVELS[@]}"; do
      log_message "Analyzing level ${level}..."
      
      if [ -f "runs/${RUN_NAME}/level_${level}_eval_summary.json" ]; then
        python3 scripts/benchmark_eval_analysis.py \
          run_name=${RUN_NAME} \
          level=${level} \
          hardware=H100 \
          baseline=baseline_time_torch \
          2>&1 | tee "${LOG_DIR}/analysis_level_${level}.log"
        
        # Copy the JSON results to our analysis directory for later processing
        cp runs/${RUN_NAME}/level_${level}_eval_summary.json runs/${RUN_NAME}/analysis/
      else
        log_message "WARNING: Summary file for level ${level} not found, skipping analysis"
      fi
    done
    
    update_checkpoint "${checkpoint_key}" "true"
    log_message "Analysis completed"
  else
    log_message "Skipping analysis (already completed)"
  fi
}

# Main execution
log_message "Starting KernelBench o4-mini benchmark"

for level in "${LEVELS[@]}"; do
  run_generation $level
  run_evaluation $level
done

run_analysis

log_message "Benchmark completed successfully"
log_message "Results are in runs/${RUN_NAME}/analysis/"

# Generate a simple report
echo "================ O4-mini KernelBench Report ================" > runs/${RUN_NAME}/report.txt
echo "Date: $(date)" >> runs/${RUN_NAME}/report.txt
echo "" >> runs/${RUN_NAME}/report.txt

for level in "${LEVELS[@]}"; do
  echo "Level ${level} Results:" >> runs/${RUN_NAME}/report.txt
  
  if [ -f "runs/${RUN_NAME}/level_${level}_eval_summary.json" ]; then
    # Extract key metrics from the JSON file
    python3 -c "
import json
import sys

try:
    with open('runs/${RUN_NAME}/level_${level}_eval_summary.json', 'r') as f:
        data = json.load(f)
    
    # Extract relevant metrics
    correct_count = data.get('correct_count', 0)
    total_count = data.get('total_count', 0)
    correctness_rate = data.get('correctness_rate', 0)
    
    # Extract fast_p metrics
    fast_metrics = {k: v for k, v in data.items() if k.startswith('fast_')}
    
    # Print to output
    print(f'Total problems: {total_count}')
    print(f'Correct solutions: {correct_count}')
    print(f'Correctness rate: {correctness_rate:.2f}')
    print('\\nPerformance metrics:')
    for metric, value in sorted(fast_metrics.items()):
        speed = metric.split('_')[1]
        print(f'  {metric}: {value:.4f} (correct & {speed}x faster than PyTorch)')
    
except Exception as e:
    print(f'Error processing results: {e}')
" >> runs/${RUN_NAME}/report.txt
  else
    echo "No results file found for level ${level}" >> runs/${RUN_NAME}/report.txt
  fi
  
  echo "" >> runs/${RUN_NAME}/report.txt
done

log_message "Report generated at runs/${RUN_NAME}/report.txt"
