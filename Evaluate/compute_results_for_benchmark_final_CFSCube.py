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

def compute_nDCG(retrieved, candidate_to_score, max_k):
    """
    retrieved: list of retrieved doc IDs
    candidate_to_score: dict {doc_id: relevance score}
    max_k: evaluate at top-k
    """

    # ---- DCG ----
    dcg = 0.0
    for i in range(1, max_k + 1):
        if i-1 >= len(retrieved):
            break

        doc_id = retrieved[i-1]
        #rel_i = candidate_to_score.get(doc_id, 0)
        rel_i = candidate_to_score[doc_id]
        dcg += rel_i / math.log2(i + 1)

    # ---- IDCG: sorted by relevance ----
    ideal_scores = sorted(candidate_to_score.values(), reverse=True)[:max_k]

    idcg = 0.0
    for i, rel in enumerate(ideal_scores, start=1):
        idcg += rel / math.log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_recall(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    relevant_at_k = set(relevant).intersection(set(retrieved_at_k))
    return len(relevant_at_k) / len(relevant) if relevant else 0


recall_50 = []
recall_30 = []
recall_20 = []
recall_15 = []
recall_10 = []
recall_5 = []
map_5 = []
map_10 = []
map_20 = []
ndcg_5 = []
ndcg_10 = []
ndcg_15 = []
ndcg_20 = []
ndcg_30 = []
ndcg_50 = []

split="result"
embedding_model = "granite-embeddings-english-r2"
baseline = False

if baseline == True:
    evaluation_template = ""
else:
    evaluation_template = ""

total_test_sets = 0
for iteration in range(1, 2):
    total_recall_50 = 0
    total_recall_30 = 0
    total_recall_20 = 0
    total_recall_15 = 0
    total_recall_10 = 0
    total_recall_5 = 0
    total_map_5 = 0
    total_map_10 = 0
    total_map_20 =0
    total_ndcg_5 = 0
    total_ndcg_10 = 0
    total_ndcg_15 = 0
    total_ndcg_20 = 0
    total_ndcg_30 = 0
    total_ndcg_50 = 0
    number_of_test_sets = 0

    current = evaluation_template.format(split=split, embedding_model=embedding_model, iteration=iteration)
    original_benchmark_path = "Existing_Benchmarks/Formatted_CSFCube/" + f"csfcube-{split}"

    file_list = traverse_folder(current)
    print(current)
    for file in file_list:
        filename = os.path.basename(file)
            
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
                        
        candidate_to_score = {}
        benchmark_directory = os.path.join(original_benchmark_path, filename)
            
        with open(benchmark_directory, "r") as json_file:
            benchmark_data = json.load(json_file)
                
        target_corpus = benchmark_data["Target_Corpus"]
            
        for corpus in target_corpus:
            formatted_candidate = f"Title: {corpus['title']}\nAbstract: {corpus['abstract']}"
            try:
                candidate_to_score[formatted_candidate]
                raise ValueError("There is duplicate in the target corpus")
            except:
                candidate_to_score[formatted_candidate] = corpus["score"]
                                
        correct_candidates = data["Correct_Candidates"]
        total_test_sets = total_test_sets + 1

        recall50 = compute_recall(final_retrieved_result, correct_candidates, 50)
        total_recall_50 = total_recall_50 + recall50
            
        recall30 = current_results["Recall@30"]
        total_recall_30 = total_recall_30 + recall30
            
        recall20 = current_results["Recall@20"]
        total_recall_20 = total_recall_20 + recall20
            
        recall15 = compute_recall(final_retrieved_result, correct_candidates, 15)
        total_recall_15 = total_recall_15 + recall15
            
        recall10 = current_results["Recall@10"]
        total_recall_10 = total_recall_10 + recall10
            
        recall5 = current_results["Recall@5"]
        total_recall_5 = total_recall_5 + recall5

        current_map_5 = compute_MAP(final_retrieved_result, correct_candidates, 5)
        total_map_5 = total_map_5 + current_map_5

        current_map_10 = compute_MAP(final_retrieved_result, correct_candidates, 10)
        total_map_10 = total_map_10 + current_map_10

        current_map_20 = compute_MAP(final_retrieved_result, correct_candidates, 20)
        total_map_20 = total_map_20 + current_map_20
            
        current_ndcg_5 = compute_nDCG(final_retrieved_result, candidate_to_score, 5)
        total_ndcg_5 = total_ndcg_5 + current_ndcg_5
            
        current_ndcg_10 = compute_nDCG(final_retrieved_result, candidate_to_score, 10)
        total_ndcg_10 = total_ndcg_10 + current_ndcg_10

        current_ndcg_15 = compute_nDCG(final_retrieved_result, candidate_to_score, 15)
        total_ndcg_15 = total_ndcg_15 + current_ndcg_15
            
        current_ndcg_20 = compute_nDCG(final_retrieved_result, candidate_to_score, 20)
        total_ndcg_20 = total_ndcg_20 + current_ndcg_20
            
        current_ndcg_30 = compute_nDCG(final_retrieved_result, candidate_to_score, 30)
        total_ndcg_30 = total_ndcg_30 + current_ndcg_30

        current_ndcg_50 = compute_nDCG(final_retrieved_result, candidate_to_score, 50)
        total_ndcg_50 = total_ndcg_50 + current_ndcg_50
                
        number_of_test_sets = number_of_test_sets + 1


    average_recall_50 = total_recall_50 / number_of_test_sets   
    average_recall_30 = total_recall_30 / number_of_test_sets
    average_recall_20 = total_recall_20 / number_of_test_sets
    average_recall_15 = total_recall_15 / number_of_test_sets
    average_recall_10 = total_recall_10 / number_of_test_sets
    average_recall_5 = total_recall_5 / number_of_test_sets
    average_map_5 = total_map_5 / number_of_test_sets
    average_map_10 = total_map_10 / number_of_test_sets
    average_map_20 = total_map_20 / number_of_test_sets
    average_ndcg_5 = total_ndcg_5 / number_of_test_sets
    average_ndcg_10 = total_ndcg_10 / number_of_test_sets
    average_ndcg_15 = total_ndcg_15 / number_of_test_sets
    average_ndcg_20 = total_ndcg_20 / number_of_test_sets
    average_ndcg_30 = total_ndcg_30 / number_of_test_sets
    average_ndcg_50 = total_ndcg_50 / number_of_test_sets

    recall_50.append(average_recall_50)
    recall_30.append(average_recall_30)
    recall_20.append(average_recall_20)
    recall_15.append(average_recall_15)
    recall_10.append(average_recall_10)
    recall_5.append(average_recall_5)
    map_5.append(average_map_5)
    map_10.append(average_map_10)
    map_20.append(average_map_20)
    ndcg_5.append(average_ndcg_5)
    ndcg_10.append(average_ndcg_10)
    ndcg_15.append(average_ndcg_15)
    ndcg_20.append(average_ndcg_20)
    ndcg_30.append(average_ndcg_30)
    ndcg_50.append(average_ndcg_50)


recall_50_mean, recall_50_std = compute_statistics(recall_50)
recall_30_mean, recall_30_std = compute_statistics(recall_30)
recall_20_mean, recall_20_std = compute_statistics(recall_20)
recall_15_mean, recall_15_std = compute_statistics(recall_15)
recall_10_mean, recall_10_std = compute_statistics(recall_10)
recall_5_mean, recall_5_std = compute_statistics(recall_5)
map_5_mean, map_5_std = compute_statistics(map_5)
map_10_mean, map_10_std = compute_statistics(map_10)
map_20_mean, map_20_std = compute_statistics(map_20)
ndcg_5_mean, ndcg_5_std = compute_statistics(ndcg_5)
ndcg_10_mean, ndcg_10_std = compute_statistics(ndcg_10)
ndcg_15_mean, ndcg_15_std = compute_statistics(ndcg_15)
ndcg_20_mean, ndcg_20_std = compute_statistics(ndcg_20)
ndcg_30_mean, ndcg_30_std = compute_statistics(ndcg_30)
ndcg_50_mean, ndcg_50_std = compute_statistics(ndcg_50)

print(f"Number of Total Test sets is {total_test_sets}")
print(f"Average Recall@50 is {recall_50_mean}, with standard deviation of {recall_50_std}")
print(f"Average Recall@30 is {recall_30_mean}, with standard deviation of {recall_30_std}")
print(f"Average Recall@20 is {recall_20_mean}, with standard deviation of {recall_20_std}")
print(f"Average Recall@15 is {recall_15_mean}, with standard deviation of {recall_15_std}")
print(f"Average Recall@10 is {recall_10_mean}, with standard deviation of {recall_10_std}")
print(f"Average Recall@5 is {recall_5_mean}, with standard deviation of {recall_5_std}")
print(f"Average MAP@5 is {map_5_mean}, with standard deviation of {map_5_std}")
print(f"Average MAP@10 is {map_10_mean}, with standard deviation of {map_10_std}")
print(f"Average MAP@20 is {map_20_mean}, with standard deviation of {map_20_std}")
print(f"Average ndcg@5 is {ndcg_5_mean}, with standard deviation of {ndcg_5_std}")
print(f"Average ndcg@10 is {ndcg_10_mean}, with standard deviation of {ndcg_10_std}")
print(f"Average ndcg@15 is {ndcg_15_mean}, with standard deviation of {ndcg_15_std}")
print(f"Average ndcg@20 is {ndcg_20_mean}, with standard deviation of {ndcg_20_std}")
print(f"Average ndcg@30 is {ndcg_30_mean}, with standard deviation of {ndcg_30_std}")
print(f"Average ndcg@50 is {ndcg_50_mean}, with standard deviation of {ndcg_50_std}")