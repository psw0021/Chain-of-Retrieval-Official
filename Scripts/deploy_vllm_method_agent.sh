#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

## write method agent model path here
method_agent_model_path="Models/Preference_Set_Llama-3.2-3B-Instruct_DPO_ref_as_gt_True_IterRet_individual_recall_True_top_k_30/meta-llama/Llama-3.2-3B-Instruct/method_agent/merged_model"
gpu_memory_utilization=1.0

port_number=8083
python3 -m vllm.entrypoints.openai.api_server \
    --model ${method_agent_model_path} \
    --dtype half \
    --port ${port_number} \
    --gpu-memory-utilization ${gpu_memory_utilization}

