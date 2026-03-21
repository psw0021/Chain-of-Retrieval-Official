import numpy as np

def precision_at_k(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    relevant_at_k = set(relevant).intersection(set(retrieved_at_k))
    return len(relevant_at_k) / k

def recall_at_k(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    relevant_at_k = set(relevant).intersection(set(retrieved_at_k))
    return len(relevant_at_k) / len(relevant) if relevant else 0


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def dcg_at_k(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    dcg = 0
    for i, doc in enumerate(retrieved_at_k):
        if doc in relevant:
            rel = 1  # binary relevance
            dcg += rel / np.log2(i + 2)  # log2(i+2) since i starts from 0
    return dcg

def ndcg_at_k(retrieved, relevant, k):
    ideal_relevant = sorted(relevant, reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal_relevant, relevant, k)
    actual_dcg = dcg_at_k(retrieved, relevant, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0


def dcg_at_k_not_binary(retrieved, relevant, doc_to_score, k):
    retrieved_at_k = retrieved[:k]
    dcg = 0
    for i, doc in enumerate(retrieved_at_k):
        if doc in relevant:
            rel = doc_to_score[doc]
            dcg += rel / np.log2(i + 2)  # log2(i+2) since i starts from 0
    
    return dcg


def ndcg_at_k_not_binary(retrieved, relevant, doc_to_score, k):
    ideal_relevant = sorted(relevant, reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal_relevant, relevant, k)
    actual_dcg = dcg_at_k(retrieved, relevant, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0


def Mean_Average_Precision(retrieved, relevant, k, max_k):
    precisions = []
    for i in range(1, max_k + 1):
        precision_at_i = precision_at_k(retrieved, relevant, i)
        precisions.append(precision_at_i)
    
    total_precision = 0
    for j in range(0, len(precisions)):
        total_precision = total_precision + precisions[j]
        
    mean_average_precision = total_precision/len(precisions)
    
    return mean_average_precision


def evaluate_retrieval(retrieved_docs, relevant_docs, k, max_k):
    #precision = precision_at_k(retrieved_docs, relevant_docs, k)
    recall300 = recall_at_k(retrieved_docs, relevant_docs, 300)
    ndcg300 = ndcg_at_k(retrieved_docs, relevant_docs, 300)
    
    recall200 = recall_at_k(retrieved_docs, relevant_docs, 200)
    ndcg200 = ndcg_at_k(retrieved_docs, relevant_docs, 200)
    
    recall100 = recall_at_k(retrieved_docs, relevant_docs, 100)
    ndcg100 = ndcg_at_k(retrieved_docs, relevant_docs, 100)
    
    
    #f1 = f1_score(precision, recall)
    #mean_average_precision = Mean_Average_Precision(retrieved_docs, relevant_docs, k, max_k)
    return {
        "Recall@300": recall300,
        "nDCG@300": ndcg300,
        "Recall@200": recall200,
        "nDCG@200": ndcg200,
        "Recall@100": recall100,
        "nDCG@100": ndcg100,
    }
    
    
def evaluate_retrieval_SciDocs(retrieved_docs, relevant_docs, doc_to_score, k, max_k):
    recall10 = recall_at_k(retrieved_docs, relevant_docs, 10)
    ndcg10 = ndcg_at_k_not_binary(retrieved_docs, relevant_docs, doc_to_score, 10)
    
    recall5 = recall_at_k(retrieved_docs, relevant_docs, 5)
    ndcg5 = ndcg_at_k_not_binary(retrieved_docs, relevant_docs, doc_to_score, 5)
    
    recall3 = recall_at_k(retrieved_docs, relevant_docs, 3)
    ndcg3 = ndcg_at_k_not_binary(retrieved_docs, relevant_docs, doc_to_score, 3)
    
    #f1 = f1_score(precision, recall)
    #mean_average_precision = Mean_Average_Precision(retrieved_docs, relevant_docs, k, max_k)
    return {
        "Recall@10": recall10,
        "nDCG@10": ndcg10,
        "Recall@5": recall5,
        "nDCG@5": ndcg5,
        "Recall@3": recall3,
        "nDCG@3": ndcg3,
    }
    
def evaluate_retrieval_SciDocsFull(retrieved_docs, relevant_docs, k, max_k):
    recall10 = recall_at_k(retrieved_docs, relevant_docs, 10)
    
    recall5 = recall_at_k(retrieved_docs, relevant_docs, 5)
    
    recall3 = recall_at_k(retrieved_docs, relevant_docs, 3)
    
    #f1 = f1_score(precision, recall)
    #mean_average_precision = Mean_Average_Precision(retrieved_docs, relevant_docs, k, max_k)
    return {
        "Recall@10": recall10,
        "Recall@5": recall5,
        "Recall@3": recall3,
    }


def evaluate_retrieval_CSFCube(retrieved_docs, relevant_docs, doc_to_score, k, max_k):
    recall30 = recall_at_k(retrieved_docs, relevant_docs, 30)
    ndcg30 = ndcg_at_k_not_binary(retrieved_docs, relevant_docs, doc_to_score, 30)
    
    recall20 = recall_at_k(retrieved_docs, relevant_docs, 20)
    ndcg20 = ndcg_at_k_not_binary(retrieved_docs, relevant_docs, doc_to_score, 20)
    
    recall10 = recall_at_k(retrieved_docs, relevant_docs, 10)
    ndcg10 = ndcg_at_k_not_binary(retrieved_docs, relevant_docs, doc_to_score, 10)
    
    recall5 = recall_at_k(retrieved_docs, relevant_docs, 5)
    ndcg5 = ndcg_at_k_not_binary(retrieved_docs, relevant_docs, doc_to_score, 5)
    
    #f1 = f1_score(precision, recall)
    #mean_average_precision = Mean_Average_Precision(retrieved_docs, relevant_docs, k, max_k)
    return {
        "Recall@30": recall30,
        "nDCG@30": ndcg30,
        "Recall@20": recall20,
        "nDCG@20": ndcg20,
        "Recall@10": recall10,
        "nDCG@10": ndcg10,
        "Recall@5": recall5,
        "nDCG@5": ndcg5,
    }
    
