#!/usr/bin/env python3
"""
Simple script to run all Level 2 KernelBench tasks with GPT-5
Based on the existing generate_and_eval_single_sample.py pattern

Usage:
    # Run all 100 Level 2 problems (requires valid OPENAI_API_KEY):
    export OPENAI_API_KEY="your-api-key-here"
    python scripts/run_all_level2_gpt5.py
    
    # Test with first 3 problems:
    python scripts/run_all_level2_gpt5.py --test

Results are saved in runs/ directory with timestamp for later analysis.
"""

import os
import sys
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

# Add src to path for imports
REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO_TOP_DIR, 'src'))

from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.utils import extract_first_code, set_gpu_arch, create_inference_server_from_presets

# Constants - all configuration at top of file
DATASET_SRC = "huggingface"
DATASET_NAME = "ScalingIntelligence/KernelBench"
LEVEL = 1
SERVER_TYPE = "openai"
MODEL_NAME = "gpt-5"
TEMPERATURE = 0.0
MAX_TOKENS = 16384
GPU_ARCH = ["Ampere"]
NUM_WORKERS = 5  # Parallel workers for generation
VERBOSE = True

# Evaluation settings
NUM_CORRECT_TRIALS = 5
NUM_PERF_TRIALS = 100
TIMEOUT_PER_PROBLEM = 300  # seconds

# Output settings
RUN_NAME = f"gpt5_level1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUNS_DIR = os.path.join(REPO_TOP_DIR, "runs")
OUTPUT_DIR = os.path.join(RUNS_DIR, RUN_NAME)

# Test subset for validation (problem IDs to test with)
TEST_SUBSET = [1, 2, 3]  # First 3 problems for testing


class TaskResult:
    """Simple container for task results"""
    def __init__(self, problem_id: int, success: bool, error: str = None, 
                 eval_result=None, generation_time: float = 0, eval_time: float = 0):
        self.problem_id = problem_id
        self.success = success
        self.error = error
        self.eval_result = eval_result
        self.generation_time = generation_time
        self.eval_time = eval_time
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        result = {
            'problem_id': self.problem_id,
            'success': self.success,
            'timestamp': self.timestamp,
            'generation_time': self.generation_time,
            'eval_time': self.eval_time
        }
        if self.error:
            result['error'] = str(self.error)  # Convert to string to ensure JSON serializable
        if self.eval_result:
            result['eval_result'] = {
                'compiled': self.eval_result.compiled,
                'correctness': self.eval_result.correctness,
                'runtime': self.eval_result.runtime,
                'runtime_stats': self._serialize_runtime_stats(self.eval_result.runtime_stats),
                'metadata': self._serialize_metadata(self.eval_result.metadata)
            }
        return result
    
    def _serialize_runtime_stats(self, stats):
        """Convert runtime stats to JSON-serializable format"""
        if stats is None:
            return None
        if isinstance(stats, dict):
            return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) 
                   for k, v in stats.items()}
        return str(stats)
    
    def _serialize_metadata(self, metadata):
        """Convert metadata to JSON-serializable format"""
        if metadata is None:
            return None
        if isinstance(metadata, dict):
            serialized = {}
            for k, v in metadata.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    serialized[k] = v
                else:
                    serialized[k] = str(v)  # Convert non-serializable objects to strings
            return serialized
        return str(metadata)


def process_single_problem(problem_id: int, dataset, inference_server) -> TaskResult:
    """Process a single problem: generate kernel and evaluate"""
    start_time = time.time()
    
    try:
        # 1. Fetch problem
        curr_problem_row = dataset.filter(lambda x: x["problem_id"] == problem_id)
        if len(curr_problem_row) == 0:
            return TaskResult(problem_id, False, f"Problem {problem_id} not found in dataset")
        
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]
        
        if VERBOSE:
            print(f"Processing Problem {problem_id}: {problem_name}")
        
        # 2. Generate kernel
        gen_start = time.time()
        custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
        custom_cuda = inference_server(custom_cuda_prompt)
        custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
        gen_time = time.time() - gen_start
        
        if custom_cuda is None:
            return TaskResult(problem_id, False, "Failed to generate valid CUDA code", 
                            generation_time=gen_time)
        
        # Save generated kernel
        kernel_path = os.path.join(OUTPUT_DIR, f"problem_{problem_id}_kernel.py")
        with open(kernel_path, "w") as f:
            f.write(custom_cuda)
        
        # 3. Evaluate kernel
        eval_start = time.time()
        eval_result = eval_kernel_against_ref(
            ref_arch_src, custom_cuda, 
            verbose=False,  # Reduce noise in parallel execution
            measure_performance=True,
            num_correct_trials=NUM_CORRECT_TRIALS,
            num_perf_trials=NUM_PERF_TRIALS
        )
        eval_time = time.time() - eval_start
        
        total_time = time.time() - start_time
        if VERBOSE:
            print(f"Problem {problem_id} completed in {total_time:.1f}s - "
                  f"Compiled: {eval_result.compiled}, Correct: {eval_result.correctness}")
        
        return TaskResult(problem_id, True, eval_result=eval_result,
                         generation_time=gen_time, eval_time=eval_time)
        
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"Error processing problem {problem_id}: {str(e)}"
        if VERBOSE:
            print(error_msg)
        return TaskResult(problem_id, False, error_msg, generation_time=total_time)


def safe_json_serialize(obj):
    """Safely serialize objects to JSON, converting non-serializable types to strings"""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    else:
        return str(obj)  # Convert anything else to string


def save_results(results: list[TaskResult], summary_stats: dict):
    """Save results to JSON files for later analysis"""
    
    try:
        # Save detailed results
        results_data = {
            'run_info': {
                'run_name': RUN_NAME,
                'timestamp': datetime.now().isoformat(),
                'level': LEVEL,
                'model': MODEL_NAME,
                'total_problems': len(results)
            },
            'config': {
                'server_type': SERVER_TYPE,
                'model_name': MODEL_NAME,
                'temperature': TEMPERATURE,
                'max_tokens': MAX_TOKENS,
                'num_correct_trials': NUM_CORRECT_TRIALS,
                'num_perf_trials': NUM_PERF_TRIALS
            },
            'results': [r.to_dict() for r in results],
            'summary': summary_stats
        }
        
        # Ensure everything is JSON serializable
        results_data = safe_json_serialize(results_data)
        
        results_path = os.path.join(OUTPUT_DIR, "detailed_results.json")
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)
        
        # Save summary for quick analysis
        summary_stats = safe_json_serialize(summary_stats)
        summary_path = os.path.join(OUTPUT_DIR, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"\nResults saved to: {OUTPUT_DIR}")
        print(f"  - Detailed: {results_path}")
        print(f"  - Summary: {summary_path}")
        
    except Exception as e:
        print(f"\nERROR: Failed to save results: {e}")
        print(f"Results data will be printed to console as backup:")
        print("="*60)
        print(f"SUMMARY: {summary_stats}")
        print(f"TOTAL RESULTS: {len(results)}")
        for i, result in enumerate(results):
            print(f"Result {i+1}: Problem {result.problem_id}, Success: {result.success}")
            if result.error:
                print(f"  Error: {str(result.error)[:100]}...")
        print("="*60)


def compute_summary_stats(results: list[TaskResult]) -> dict:
    """Compute summary statistics for analysis"""
    total = len(results)
    successful = sum(1 for r in results if r.success)
    
    # For successful evaluations
    eval_results = [r for r in results if r.success and r.eval_result]
    compiled = sum(1 for r in eval_results if r.eval_result.compiled)
    correct = sum(1 for r in eval_results if r.eval_result.correctness)
    
    # Timing stats
    gen_times = [r.generation_time for r in results if r.generation_time > 0]
    eval_times = [r.eval_time for r in results if r.eval_time > 0]
    
    # Performance stats (speedups)
    speedups = []
    for r in eval_results:
        if (r.eval_result.correctness and r.eval_result.runtime and 
            hasattr(r.eval_result, 'runtime_stats') and r.eval_result.runtime_stats):
            ref_time = r.eval_result.runtime_stats.get('ref_time_mean')
            custom_time = r.eval_result.runtime_stats.get('custom_time_mean')
            if ref_time and custom_time and custom_time > 0:
                speedups.append(ref_time / custom_time)
    
    summary = {
        'total_problems': total,
        'successful_generations': successful,
        'compiled_kernels': compiled,
        'correct_kernels': correct,
        'success_rate': successful / total if total > 0 else 0,
        'compilation_rate': compiled / successful if successful > 0 else 0,
        'correctness_rate': correct / successful if successful > 0 else 0,
        'fast_0': correct / total if total > 0 else 0,  # Fraction correct (fast_0 metric)
    }
    
    if gen_times:
        summary['avg_generation_time'] = sum(gen_times) / len(gen_times)
    if eval_times:
        summary['avg_eval_time'] = sum(eval_times) / len(eval_times)
    if speedups:
        summary['avg_speedup'] = sum(speedups) / len(speedups)
        summary['fast_1'] = sum(1 for s in speedups if s > 1.0) / total  # Faster than PyTorch
        summary['fast_2'] = sum(1 for s in speedups if s > 2.0) / total  # 2x faster
    
    return summary


def main():
    """Main execution function"""
    print(f"Starting KernelBench Level {LEVEL} evaluation with {MODEL_NAME}")
    print(f"Run name: {RUN_NAME}")
    
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_gpu_arch(GPU_ARCH)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    curr_level_dataset = dataset[f"level_{LEVEL}"]
    total_problems = len(curr_level_dataset)
    print(f"Found {total_problems} problems in Level {LEVEL}")
    
    # Create inference server
    print(f"Setting up {MODEL_NAME} inference...")
    inference_server = create_inference_server_from_presets(
        server_type=SERVER_TYPE,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        verbose=VERBOSE,
        time_generation=True
    )
    
    # Determine which problems to run
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        problem_ids = TEST_SUBSET
        print(f"Running test subset: {problem_ids}")
    else:
        problem_ids = list(range(1, total_problems + 1))
        print(f"Running all {len(problem_ids)} problems")
    
    # Process problems in parallel
    print(f"Processing with {NUM_WORKERS} parallel workers...")
    results = []
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        future_to_problem = {
            executor.submit(process_single_problem, pid, curr_level_dataset, inference_server): pid
            for pid in problem_ids
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_problem, timeout=TIMEOUT_PER_PROBLEM * len(problem_ids)):
            try:
                result = future.result(timeout=TIMEOUT_PER_PROBLEM)
                results.append(result)
            except Exception as e:
                problem_id = future_to_problem[future]
                print(f"Problem {problem_id} failed with exception: {e}")
                results.append(TaskResult(problem_id, False, str(e)))
    
    # Sort results by problem_id
    results.sort(key=lambda x: x.problem_id)
    
    # Compute and display summary
    summary_stats = compute_summary_stats(results)
    
    print(f"\n{'='*60}")
    print("SUMMARY RESULTS")
    print(f"{'='*60}")
    print(f"Total Problems: {summary_stats['total_problems']}")
    print(f"Successful Generations: {summary_stats['successful_generations']}")
    print(f"Compiled Kernels: {summary_stats['compiled_kernels']}")
    print(f"Correct Kernels: {summary_stats['correct_kernels']}")
    print(f"Success Rate: {summary_stats['success_rate']:.2%}")
    print(f"Compilation Rate: {summary_stats['compilation_rate']:.2%}")
    print(f"Correctness Rate (fast_0): {summary_stats['fast_0']:.2%}")
    
    if 'avg_speedup' in summary_stats:
        print(f"Average Speedup: {summary_stats['avg_speedup']:.2f}x")
        print(f"Faster than PyTorch (fast_1): {summary_stats['fast_1']:.2%}")
        print(f"2x faster (fast_2): {summary_stats['fast_2']:.2%}")
    
    if 'avg_generation_time' in summary_stats:
        print(f"Avg Generation Time: {summary_stats['avg_generation_time']:.1f}s")
    if 'avg_eval_time' in summary_stats:
        print(f"Avg Evaluation Time: {summary_stats['avg_eval_time']:.1f}s")
    
    # Save results
    save_results(results, summary_stats)
    
    print(f"\nRun completed: {RUN_NAME}")


if __name__ == "__main__":
    main()
