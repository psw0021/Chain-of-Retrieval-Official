#!/bin/bash
use_full_paper_as_query=True
benchmark_directory="Paper2PaperRetrievalBench/SciFullBench/Final_Dataset_w_citations_mentions_removed/Benchmark/ACL/Cited_Papers"
embedding_model="jina-embeddings-v2-base-en"
top_k=300
max_top_k=300
author_data_incorporation=False
use_introduction_as_query=True
use_introduction_as_corpus=True
corpus_directory="Paper2PaperRetrievalBench/SciFullBench/Final_Dataset_w_citations_mentions_removed/Target_Corpus/target_corpus_citations_removed_True_mentions_removed_True/corpus.json"
batch_size=1
use_full_paper_as_corpus=True
remove_citations=True 
remove_mentions=True


function run_job() {
    local start_idx=$1
    local end_idx=$2
    local job_num=$3
    local iteration=$4
    echo "Running inference_metadata_retrieval.py"
    rm logs/logs_metadata/output_${job_num}.log 
    python Inference/inference_metadata_retrieval.py \
    --iteration_num ${iteration}    \
    --batch_size ${batch_size} \
    --use_full_paper_as_query ${use_full_paper_as_query} \
    --corpus_directory ${corpus_directory} \
    --start_idx ${start_idx} \
    --end_idx ${end_idx} \
    --benchmark_directory ${benchmark_directory} \
    --author_data_incorporation ${author_data_incorporation}    \
    --use_introduction_as_query ${use_introduction_as_query} \
    --use_introduction_as_corpus ${use_introduction_as_corpus} \
    --remove_citations ${remove_citations}  \
    --remove_mentions ${remove_mentions}    \
    --embedding_model ${embedding_model} \
    --top_k ${top_k} \
    --max_top_k ${max_top_k} \
    --use_full_paper_as_corpus ${use_full_paper_as_corpus}  > logs/logs_metadata/output_${job_num}.log 2>&1
}




for i in {1..1}; do
    echo "Iteration $i"
    iteration=$i
    current_start=0  # Reset for each iteration
        
    total_indices=400
    indices_per_thread=400
    export CUDA_VISIBLE_DEVICES=4

    job_count=0
    while [ "$current_start" -lt "$total_indices" ]; do
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