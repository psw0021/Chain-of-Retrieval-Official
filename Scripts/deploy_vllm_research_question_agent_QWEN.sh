#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

## write research question agent model path here(trained with QWEN-2.5-3B-Instruct model)
research_question_agent_model_path="Models/Preference_Set_Qwen2.5-3B-Instruct_JEmb_ref_as_gt_True_IterRet_individual_recall_True_top_k_30/Qwen/Qwen2.5-3B-Instruct/research_question_agent/merged_model"
gpu_memory_utilization=1.0
max_model_len=131072

port_number=8082
python3 -m vllm.entrypoints.openai.api_server \
    --model ${research_question_agent_model_path} \
    --dtype half \
    --port ${port_number} \
    --gpu-memory-utilization ${gpu_memory_utilization}  \
    --rope_scaling '{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}' \
    --max_model_len ${max_model_len}

