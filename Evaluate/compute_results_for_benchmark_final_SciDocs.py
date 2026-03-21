import os
import json
import numpy as np
import scipy.stats as stats
import sys

def traverse_folder(folder_path):
    """
    Traverse through the folder and retrieve file names.

    Args:
        folder_path (str): Path to the folder to traverse.

    Returns:
        list: A list of file paths.
    """
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def return_immediate_subdirectory(directory):
    subdirs = [entry.path for entry in os.scandir(directory) if entry.is_dir()]

    return subdirs

def compute_statistics(data):
    N = len(data)
    mean = np.mean(data)                         # Compute mean
    std_dev = np.std(data, ddof=1)               # Compute sample standard deviation

    return mean, std_dev

def compute_mrr(ranked_lists, ground_truths, top_k):
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        ranked_lists (List[Any]]]): A list of ranked retrieval results
        ground_truths (List[Any]): A list of the correct item (or set of correct items) for each query.

    Returns:
        float: The MRR score.
    """

    ranked_lists = ranked_lists[:top_k]

    rank = 0
    
    for i, item in enumerate(ranked_lists, start=1):
        if item in ground_truths:
            rank = i
            rank_inverse = 1/rank
            
            return rank_inverse
    
    return 0

def precision_at_k(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    relevant_at_k = set(relevant).intersection(set(retrieved_at_k))
    
    return len(relevant_at_k) / k

def compute_MAP(retrieved, relevant, max_k):
    num_relevant = 0
    precisions = []

    for i in range(1, max_k + 1):
        if retrieved[i-1] in relevant:
            num_relevant += 1
            precision_at_i = num_relevant / i
            precisions.append(precision_at_i)

    if len(precisions) == 0:
        return 0.0  

    mean_average_precision = sum(precisions) / len(relevant)
    return mean_average_precision

import math

def compute_NDCG_binary(retrieved, relevant, max_k):
    """
    retrieved: list of retrieved document IDs
    relevant: set of relevant document IDs
    max_k: evaluate at top-k
    """
    dcg = 0.0
    for i in range(1, max_k + 1):
        if i-1 >= len(retrieved):
            break
        doc_id = retrieved[i-1]
        rel_i = 1 if doc_id in relevant else 0
        dcg += (rel_i) / math.log2(i + 1)

    # Ideal DCG 
    ideal_relevances = [1] * min(len(relevant), max_k) 
    idcg = 0.0
    for i in range(1, len(ideal_relevances) + 1):
        idcg += (1) / math.log2(i + 1)

    if idcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


def compute_recall(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    relevant_at_k = set(relevant).intersection(set(retrieved_at_k))
    return len(relevant_at_k) / len(relevant) if relevant else 0

   

recall_10 = []
recall_5 = []
recall_3 = []
recall_1 = []
mrr_5 = []
map_10 = []
map_5 = []
map_3 = []
ndcg_3 = []
ndcg_5 = []
ndcg_10 = []


total_test_sets = 0
total_recall_10 = 0
total_recall_5 = 0
total_recall_3 = 0
total_recall_1 = 0
total_Mrr_5 = 0
total_map_10 = 0
total_map_5 = 0
total_map_3 = 0
total_ndcg_3 = 0
total_ndcg_5 = 0
total_ndcg_10 = 0
number_of_test_sets = 0

current = ""

file_list = traverse_folder(current)
print(current)
for file in file_list:
    with open(file, "r") as json_file:
        try:
                data = json.load(json_file)
        except:
                print(file)
                sys.exit()
    try:
            current_results = data["Current Result"]
    except:
            continue
    try:
            final_retrieved_result = data["Final_Ranked_Results"]
    except:
            retrieved_result = data["Retrieved_Candidates"]
            retrieved_result_keys = retrieved_result.keys()
            final_retrieved_result = []
            for result_key in retrieved_result_keys:
                final_retrieved_result.append(retrieved_result[result_key]["Content"])
                
        
    correct_candidates = data["Correct_Candidates"]
        
    total_test_sets = total_test_sets + 1
    
    recall10 = current_results["Recall@10"]
    total_recall_10 = total_recall_10 + recall10
    
    recall5 = current_results["Recall@5"]
    total_recall_5 = total_recall_5 + recall5
    
    recall3 = current_results["Recall@3"]
    total_recall_3 = total_recall_3 + recall3
    
    recall1 = compute_recall(final_retrieved_result, correct_candidates, 1)
    total_recall_1 = total_recall_1 + recall1

    current_mrr_5 = compute_mrr(final_retrieved_result, correct_candidates, 5)
    total_Mrr_5 = total_Mrr_5 + current_mrr_5
    
    current_map_3 = compute_MAP(final_retrieved_result, correct_candidates, 3)
    total_map_3 = total_map_3 + current_map_3

    current_map_5 = compute_MAP(final_retrieved_result, correct_candidates, 5)
    total_map_5 = total_map_5 + current_map_5
    
    current_map_10 = compute_MAP(final_retrieved_result, correct_candidates, 10)
    total_map_10 = total_map_10 + current_map_10

    current_ndcg_3 = compute_NDCG_binary(final_retrieved_result, correct_candidates, 3)
    total_ndcg_3 = total_ndcg_3 + current_ndcg_3

    current_ndcg_5 = compute_NDCG_binary(final_retrieved_result, correct_candidates, 5)
    total_ndcg_5 = total_ndcg_5 + current_ndcg_5

    current_ndcg_10 = compute_NDCG_binary(final_retrieved_result, correct_candidates, 10)
    total_ndcg_10 = total_ndcg_10 + current_ndcg_10
        
    number_of_test_sets = number_of_test_sets + 1

    

average_recall_10 = total_recall_10 / number_of_test_sets
average_recall_5 = total_recall_5 / number_of_test_sets
average_recall_3 = total_recall_3 / number_of_test_sets
average_recall_1 = total_recall_1 / number_of_test_sets
average_mrr_5 = total_Mrr_5 / number_of_test_sets
average_map_3 = total_map_3 / number_of_test_sets
average_map_5 = total_map_5 / number_of_test_sets
average_map_10 = total_map_10 / number_of_test_sets
average_ndcg_3 = total_ndcg_3 / number_of_test_sets
average_ndcg_5 = total_ndcg_5 / number_of_test_sets
average_ndcg_10 = total_ndcg_10 / number_of_test_sets

recall_10.append(average_recall_10)
recall_5.append(average_recall_5)
recall_3.append(average_recall_3)
recall_1.append(average_recall_1)
mrr_5.append(average_mrr_5)
map_3.append(average_map_3)
map_5.append(average_map_5)
map_10.append(average_map_10)
ndcg_3.append(average_ndcg_3)
ndcg_5.append(average_ndcg_5)
ndcg_10.append(average_ndcg_10)


recall_10_mean, recall_10_std = compute_statistics(recall_10)
recall_5_mean, recall_5_std = compute_statistics(recall_5)
recall_3_mean, recall_3_std = compute_statistics(recall_3)
recall_1_mean, recall_1_std = compute_statistics(recall_1)
mrr_5_mean, mrr_5_std = compute_statistics(mrr_5)
map_3_mean, map_3_std = compute_statistics(map_3)
map_5_mean, map_5_std = compute_statistics(map_5)
map_10_mean, map_10_std = compute_statistics(map_10)
ndcg_3_mean, ndcg_3_std = compute_statistics(ndcg_3)
ndcg_5_mean, ndcg_5_std = compute_statistics(ndcg_5)
ndcg_10_mean, ndcg_10_std = compute_statistics(ndcg_10)

print(f"Number of Total Test sets is {total_test_sets}")
print(f"Average Recall@10 is {recall_10_mean}, with standard deviation of {recall_10_std}")
print(f"Average Recall@5 is {recall_5_mean}, with standard deviation of {recall_5_std}")
print(f"Average Recall@3 is {recall_3_mean}, with standard deviation of {recall_3_std}")
print(f"Average Recall@1 is {recall_1_mean}, with standard deviation of {recall_1_std}")
print(f"Average MRR@5 is {mrr_5_mean}, with standard deviation of {mrr_5_std}")
print(f"Average MAP@3 is {map_3_mean}, with standard deviation of {map_3_std}")
print(f"Average MAP@5 is {map_5_mean}, with standard deviation of {map_5_std}")
print(f"Average MAP@10 is {map_10_mean}, with standard deviation of {map_10_std}")
print(f"Average ndcg@3 is {ndcg_3_mean}, with standard deviation of {ndcg_3_std}")
print(f"Average ndcg@5 is {ndcg_5_mean}, with standard deviation of {ndcg_5_std}")
print(f"Average ndcg@10 is {ndcg_10_mean}, with standard deviation of {ndcg_10_std}")