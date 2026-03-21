#!/bin/bash
echo $CUDA_VISIBLE_DEVICES

# conda environment
CONDA_ENV_NAME="paper_retrieval"
### Environmental variables

use_query_optimizer=False
multi_agent=False
query_optimizer_model="gpt-4.1-2025-04-14"
use_abstract_for_query_optimization=False
gpu_memory_utilization=0.7
use_gpt=True
vllm_api_truncation=False
## Set the benchmark directory for SciFullBench, as shown below
benchmark_directory="Existing_Benchmarks/Formatted_CSFCube/csfcube-result"
embedding_model="Qwen3-Embedding-0.6B"
top_k=50
max_top_k=50
hyperparameter_RRF=60
batch_size=10
include_original_retrieval=False
use_base_agent=False
use_method_agent=False
use_experiment_agent=False
use_research_question_agent=False


iterative_retrieval=False
SubTreeSearch=False
Recursive_Merge=False
use_aspect_aware_cache_for_selection=False
total_iterative_retrieval_loop=2
starting_iteration_of_A2A_only=-1
input_top_k_for_verifier=1
selector_starting_idx=0


function run_job() {
    local start_idx=$1
    local end_idx=$2
    local job_num=$3
    local iteration=$4
    echo "Running inference_retrieval_CSFCube.py"
    rm logs/logs_CSFCube/output_${job_num}.log 
    python Inference/inference_retrieval_CSFCube.py \
    --iteration_num ${iteration}    \
    --batch_size ${batch_size} \
    --use_query_optimizer ${use_query_optimizer} \
    --multi_agent ${multi_agent}    \
    --query_optimizer_model ${query_optimizer_model} \
    --use_abstract_for_query_optimization ${use_abstract_for_query_optimization} \
    --gpu_memory_utilization ${gpu_memory_utilization}  \
    --use_gpt ${use_gpt} \
    --vllm_api_truncation ${vllm_api_truncation} \
    --start_idx ${start_idx} \
    --end_idx ${end_idx} \
    --include_original_retrieval ${include_original_retrieval} \
    --use_base_agent ${use_base_agent}  \
    --use_method_agent ${use_method_agent} \
    --use_experiment_agent ${use_experiment_agent} \
    --use_research_question_agent ${use_research_question_agent} \
    --benchmark_directory ${benchmark_directory} \
    --embedding_model ${embedding_model} \
    --top_k ${top_k} \
    --max_top_k ${max_top_k} \
    --hyperparameter_RRF ${hyperparameter_RRF} \
    --iterative_retrieval ${iterative_retrieval}    \
    --Recursive_Merge ${Recursive_Merge} \
    --use_aspect_aware_cache_for_selection ${use_aspect_aware_cache_for_selection} \
    --SubTreeSearch ${SubTreeSearch} \
    --total_iterative_retrieval_loop ${total_iterative_retrieval_loop}  \
    --starting_iteration_of_A2A_only ${starting_iteration_of_A2A_only} \
    --input_top_k_for_verifier ${input_top_k_for_verifier} \
    --selector_starting_idx ${selector_starting_idx} > logs/logs_CSFCube/output_${job_num}.log 2>&1
}


for i in {1..3}; do
    echo "Iteration $i"
    iteration=$i
    current_start=0  # Reset for each iteration
        
    total_indices=20
    indices_per_thread=5

    job_count=0
    while [ "$current_start" -lt "$total_indices" ]; do
        if [ "$job_count" -eq 0 ]; then
            export CUDA_VISIBLE_DEVICES=2
        elif [ "$job_count" -eq 1 ]; then
            export CUDA_VISIBLE_DEVICES=2
        elif [ "$job_count" -eq 2 ]; then
            export CUDA_VISIBLE_DEVICES=2
        elif [ "$job_count" -eq 3 ]; then
            export CUDA_VISIBLE_DEVICES=2
        elif [ "$job_count" -eq 4 ]; then
            export CUDA_VISIBLE_DEVICES=2
        elif [ "$job_count" -eq 5 ]; then
            export CUDA_VISIBLE_DEVICES=5
        elif [ "$job_count" -eq 6 ]; then
            export CUDA_VISIBLE_DEVICES=5
        elif [ "$job_count" -eq 7 ]; then
            export CUDA_VISIBLE_DEVICES=5
        elif [ "$job_count" -eq 8 ]; then
            export CUDA_VISIBLE_DEVICES=5
        elif [ "$job_count" -eq 9 ]; then
            export CUDA_VISIBLE_DEVICES=5
        else
            echo "Invalid JOB_ID: $job_count"
            exit 1
        fi

        current_end=$((current_start + indices_per_thread))
        if [ "$current_end" -gt "$total_indices" ]; then
            current_end=$total_indices
        fi

        # Run the job in background
        run_job $current_start $current_end $job_count $iteration &

        ((job_count++))
        current_start=$current_end
    done

    # Wait for all background jobs in this iteration to finish
    wait
done
