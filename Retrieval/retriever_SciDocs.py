import torch
import faiss
import os
import sys
from collections import namedtuple
import numpy as np
import torch
from torch import nn as nn
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from adapters import AutoAdapterModel
import json
import torch
from torch import nn as nn
from pathlib import Path
import logging
from openai import OpenAI
import tiktoken
import re
from vllm import LLM
from scipy.spatial.distance import cdist
import datasets

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
# Add the root directory to Python's module search path
sys.path.append(root_dir)
from Utils.utils import split_paragraph, reformat_to_string
from Agents.QueryOptimizer import QueryOptimizer
from Agents.Selector import Selector
from Retrieval.metrics import evaluate_retrieval_SciDocs

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your GPU setup.")

logging.basicConfig(level=logging.INFO)


class Tree_Node:
    def __init__(self, current_node_name, parent_node_name, retrieved_result):
        """
        Tree Node for saving parents, children, queries, and their retrieved result
        """
        self.current_node_name = current_node_name
        self.parent = parent_node_name
        self.retrieved_result = retrieved_result
        self.method_wise_child = None
        self.experiment_wise_child = None
        self.research_question_wise_child = None


class Retriever:
    def __init__(self, args, result_folder_path):
        """
        Our overall retriever class that retrieves input paper from massive target paper corpus.
        """
        self.benchmark_directory = args.benchmark_directory
        self.embedding_model = args.embedding_model
        self.use_gpu = torch.cuda.is_available()
        self.embedding_model_max_length = None

        self.iterative_retrieval = args.iterative_retrieval
        self.SubTreeSearch = args.SubTreeSearch
        
        self.total_iterative_retrieval_loop = args.total_iterative_retrieval_loop
    
        self.use_gpt = args.use_gpt

        self.include_original_retrieval = args.include_original_retrieval

        self.number_of_total_queries = 0
        if args.use_base_agent == True:
            self.number_of_total_queries += 3

            if self.include_original_retrieval == True:
                self.number_of_total_queries += 1
        
        elif args.use_base_agent == False:
            if args.use_method_agent == True:
                self.number_of_total_queries += 1
            if args.use_experiment_agent == True:
                self.number_of_total_queries += 1
            if args.use_research_question_agent == True:
                self.number_of_total_queries += 1
            if self.include_original_retrieval == True:
                self.number_of_total_queries += 1
        
        
        self.candidate_embeddings = None
        
        self.embedding_dimension = None
        self.use_query_optimizer = args.use_query_optimizer
        self.batch_size = args.batch_size
        self.only_A2A = False

        self.hyperparameter_RRF = args.hyperparameter_RRF
        
        self.vllm_dict_for_query_optimizer = {}
        num_devices = torch.cuda.device_count()
        
        device_list = []
        for device_number in range(num_devices):
            device_list.append(f"cuda:{device_number}")

        assert len(device_list) > 0, f"Expected at least 1 devices, but found {len(device_list)}"
        if self.use_gpt == False: 
            if self.use_query_optimizer == True and self.use_trained_model == False:
                assert len(device_list) >= 1, f"Expected at least 1 device, but found {len(device_list)}"
                if args.query_optimizer_model != "Qwen/Qwen2.5-3B-Instruct":
                    base_model = LLM(model=args.query_optimizer_model, tensor_parallel_size=1, gpu_memory_utilization=args.gpu_memory_utilization, dtype="half", device=device_list[0])
                elif args.query_optimizer_model == "Qwen/Qwen2.5-3B-Instruct":
                    base_model = LLM(model=args.query_optimizer_model, tensor_parallel_size=1, max_model_len=131072, rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}, gpu_memory_utilization=args.gpu_memory_utilization, dtype="half", device=device_list[0])
                
                self.vllm_dict_for_query_optimizer["METHOD"] = {"agent": base_model, "device": device_list[0]}
                self.vllm_dict_for_query_optimizer["EXPERIMENT"] = {"agent": base_model, "device": device_list[0]}
                self.vllm_dict_for_query_optimizer["RESEARCH_QUESTION"] = {"agent": base_model, "device": device_list[0]}
            
        self.cuda_embedding_model_device = device_list[-1]

        if self.use_query_optimizer == True:
            self.QueryOptimizer = QueryOptimizer(args, result_folder_path, self.vllm_dict_for_query_optimizer)
            
        if self.iterative_retrieval == True and self.SubTreeSearch == True:
            self.Selector = Selector(args)
        
        if self.embedding_model == "jina-embeddings-v2-base-en":
            self.question_encoder = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True, max_length=8192).to(self.cuda_embedding_model_device)
            self.context_encoder = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True, max_length=8192).to(self.cuda_embedding_model_device)
            
            self.embedding_model_max_length=8192
            self.embedding_dimension = 768
            
        elif self.embedding_model == "bge-m3":
            embedding_model = SentenceTransformer("BAAI/bge-m3")
            embedding_model.tokenizer.model_max_length = 8192

            self.question_encoder = embedding_model
            self.context_encoder = embedding_model

            logging.info(f"{embedding_model.tokenizer.model_max_length}")
            self.embedding_model_max_length = 8192
            self.embedding_dimension = 1024
            
            
        elif self.embedding_model == "inf-retriever-v1-1.5b":
            embedding_model = SentenceTransformer("infly/inf-retriever-v1-1.5b" , trust_remote_code=True)
            embedding_model.tokenizer.model_max_length = 32768

            self.question_encoder = embedding_model
            self.context_encoder = embedding_model

            logging.info(f"{embedding_model.tokenizer.model_max_length}")
            self.embedding_model_max_length = 32768
            self.embedding_dimension = 1536
            
        elif self.embedding_model == "granite-embedding-english-r2":
            embedding_model = SentenceTransformer("ibm-granite/granite-embedding-english-r2" , trust_remote_code=True)
            embedding_model.tokenizer.model_max_length = 8192

            self.question_encoder = embedding_model
            self.context_encoder = embedding_model

            logging.info(f"{embedding_model.tokenizer.model_max_length}")
            self.embedding_model_max_length = 8192
            self.embedding_dimension = 768

        elif self.embedding_model == "dewey_en_beta":
            self.embedding_model_max_length = 131072
            self.embedding_dimension = 2048
            embedding_model = SentenceTransformer(
                "infgrad/dewey_en_beta",
                trust_remote_code=True,
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                },
                config_kwargs={"single_vector_type": "mean"}
            ).bfloat16().eval()
            
            embedding_model.tokenizer.model_max_length = self.embedding_model_max_length

            self.question_encoder = embedding_model
            self.context_encoder = embedding_model

            logging.info(f"{embedding_model.tokenizer.model_max_length}")

            
        elif self.embedding_model == "SPECTER":
            tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
            model = AutoModel.from_pretrained("allenai/specter", device_map="auto").to(self.cuda_embedding_model_device)
            
            self.embedding_model_max_length=512
            self.embedding_dimension = 768
            self.question_encoder = model
            self.question_tokenizer = tokenizer
            
            self.context_encoder = model
            self.context_tokenizer = tokenizer
            
        elif self.embedding_model == "SPECTER2_Base":
            tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')  
            model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
            specter2_model = model.to(self.cuda_embedding_model_device)
            
            self.embedding_model_max_length=512
            self.embedding_dimension = 768
            
            self.question_encoder = specter2_model
            self.question_tokenizer = tokenizer
            
            self.context_encoder = specter2_model
            self.context_tokenizer = tokenizer
            
        elif self.embedding_model == "SPECTER2":
            tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')  
            model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
            model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
            specter2_model = model.to(self.cuda_embedding_model_device)
            
            self.embedding_model_max_length=512
            self.embedding_dimension = 768
            
            self.question_encoder = specter2_model
            self.question_tokenizer = tokenizer
            
            self.context_encoder = specter2_model
            self.context_tokenizer = tokenizer

        elif self.embedding_model == "SciNCL":
            tokenizer = AutoTokenizer.from_pretrained('malteos/scincl')
            model = AutoModel.from_pretrained('malteos/scincl')
            model = model.to(self.cuda_embedding_model_device)
            
            self.embedding_model_max_length=512
            self.embedding_dimension = 768
            
            self.question_encoder = model
            self.question_tokenizer = tokenizer
            
            self.context_encoder = model
            self.context_tokenizer = tokenizer

    
    def format_corpus(self, candidates):
        formatted_total_corpus = []
        formatted_total_corpus_dict = {}
        formatted_total_corpus_to_score = {}
        abstract_to_full_context_dict = {}
        for paper in candidates:
            paper_title = paper["title"]
            paper_abstract = paper["abstract"]
                    
            formatted_total_candidate = f"Title: {paper_title}\nAbstract: {paper_abstract}"
            
            formatted_total_corpus.append(formatted_total_candidate)
                
            formatted_total_corpus_to_score[formatted_total_candidate] = paper["score"]
            
            abstract_to_full_context_dict[formatted_total_candidate] = formatted_total_candidate
                    
        return formatted_total_corpus, formatted_total_corpus_to_score, abstract_to_full_context_dict
        
    
    def format_query_candidates(self, paper):
        if self.use_query_optimizer == False:
            query_paper = paper["Query"]
            query_title = query_paper["title"]
            query_abstract = query_paper["abstract"]
            
            formatted_query = f"Title: {query_title}\nAbstract: {query_abstract}"
            
        elif self.use_query_optimizer == True:
            query_paper = paper["Query"]
            query_title = query_paper["title"]
            optimized_queries = query_paper["optimized_queries"]
            formatted_query = []

            if self.include_original_retrieval == True:
                query_abstract = query_paper["abstract"]
                formatted_query.append(("Original Retrieval", query_abstract))

            for agent_name, query in optimized_queries:
                formatted_optimized_query = f"{query}"
                formatted_query.append((agent_name, formatted_optimized_query))
        
        formatted_candidates = []
        candidate_dictionary = {}
        candidate_papers = paper["Candidate"]
        for candidate in candidate_papers:
            candidate_title = candidate["title"]
            candidate_abstract = candidate["abstract"]
            
            formatted_candidate = f"Title: {candidate_title}\nAbstract: {candidate_abstract}"
            
            candidate_dictionary[formatted_candidate] = candidate["score"]
            
            formatted_candidates.append(formatted_candidate)
        
        return formatted_query, formatted_candidates  
    
        
    # Function to encode passages in batches
    def encode_passages(self, passages):
        if self.embedding_model == "jina-embeddings-v2-base-en":
            encoded_embeddings = []
            with torch.no_grad():
                embeddings = self.context_encoder.encode(passages, device=self.cuda_embedding_model_device)
            encoded_embeddings.append(embeddings)
            return np.vstack(encoded_embeddings)
        
        
        elif self.embedding_model == "bge-m3":
            encoded_embeddings = []
            with torch.no_grad():
                embeddings = self.context_encoder.encode(passages, device=self.cuda_embedding_model_device)
            encoded_embeddings.append(embeddings)
            return np.vstack(encoded_embeddings)
        
        
        elif self.embedding_model == "inf-retriever-v1-1.5b":
            encoded_embeddings = []
            with torch.no_grad():
                ## format passages
                embeddings = self.context_encoder.encode(passages, device=self.cuda_embedding_model_device)
            encoded_embeddings.append(embeddings)
            
            return np.vstack(encoded_embeddings)
        
        
        elif self.embedding_model == "SPECTER":
            encoded_embeddings = []
            inputs = self.context_tokenizer(passages, return_tensors="pt", padding=True, truncation=True, max_length=self.embedding_model_max_length).to(self.cuda_embedding_model_device)
            context_output = self.context_encoder(**inputs)
            with torch.no_grad():
                    embeddings = context_output.last_hidden_state[:, 0, :].cpu().detach().numpy()
            encoded_embeddings.append(embeddings)
            
            return np.vstack(encoded_embeddings)
        
        
        elif self.embedding_model == "SPECTER2_Base":
            encoded_embeddings = []
            inputs = self.context_tokenizer(passages, return_tensors="pt", padding=True, truncation=True, max_length=self.embedding_model_max_length).to(self.cuda_embedding_model_device)
            context_output = self.context_encoder(**inputs)
            with torch.no_grad():
                embeddings = context_output.last_hidden_state[:, 0, :].cpu().detach().numpy()
            encoded_embeddings.append(embeddings)
            
            return np.vstack(encoded_embeddings)
        
        
        elif self.embedding_model == "SPECTER2":
            encoded_embeddings = []
            inputs = self.context_tokenizer(passages, return_tensors="pt", padding=True, truncation=True, max_length=self.embedding_model_max_length).to(self.cuda_embedding_model_device)
            context_output = self.context_encoder(**inputs)
            with torch.no_grad():
                embeddings = context_output.last_hidden_state[:, 0, :].cpu().detach().numpy()
            encoded_embeddings.append(embeddings)
            
            return np.vstack(encoded_embeddings)
        
        
        elif self.embedding_model == "SciNCL":
            encoded_embeddings = []
            inputs = self.context_tokenizer(passages, return_tensors="pt", padding=True, truncation=True, max_length=self.embedding_model_max_length).to(self.cuda_embedding_model_device)
            context_output = self.context_encoder(**inputs)
            with torch.no_grad():
                embeddings = context_output.last_hidden_state[:, 0, :].cpu().detach().numpy()
            encoded_embeddings.append(embeddings)
            
            return np.vstack(encoded_embeddings)


    # Function to encode a query
    def encode_query(self, query):
        if self.embedding_model == "jina-embeddings-v2-base-en":
            with torch.no_grad():
                embedding = self.question_encoder.encode(query, device=self.cuda_embedding_model_device)  # Encode on GPU
            
            embedding = embedding.reshape(1,-1)
            return embedding
    
        
        elif self.embedding_model == "bge-m3":
            with torch.no_grad():
                embedding = self.question_encoder.encode(query, device=self.cuda_embedding_model_device)
            embedding = embedding.reshape(1, -1)
            
            return embedding
        
        
        elif self.embedding_model == "inf-retriever-v1-1.5b":
            with torch.no_grad():
                embedding = self.question_encoder.encode(query, device=self.cuda_embedding_model_device, prompt_name="query")
            embedding = embedding.reshape(1, -1)
            
            return embedding
        
        
        elif self.embedding_model == "SPECTER":
            inputs = self.question_tokenizer(query, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=self.embedding_model_max_length).to(self.cuda_embedding_model_device)
            with torch.no_grad():
                query_output = self.question_encoder(**inputs)
                embedding = query_output.last_hidden_state[:, 0, :].cpu().detach().numpy()
                
            return embedding
        
        elif self.embedding_model == "SPECTER2_Base":
            inputs = self.question_tokenizer(query, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=self.embedding_model_max_length).to(self.cuda_embedding_model_device)
            with torch.no_grad():
                query_output = self.question_encoder(**inputs)
                embedding = query_output.last_hidden_state[:, 0, :].cpu().detach().numpy()

            return embedding
        
        elif self.embedding_model == "SPECTER2":
            inputs = self.question_tokenizer(query, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=self.embedding_model_max_length).to(self.cuda_embedding_model_device)
            with torch.no_grad():
                query_output = self.question_encoder(**inputs)
                embedding = query_output.last_hidden_state[:, 0, :].cpu().detach().numpy()

            return embedding
        
        elif self.embedding_model == "SciNCL":
            inputs = self.question_tokenizer(query, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=self.embedding_model_max_length).to(self.cuda_embedding_model_device)
            with torch.no_grad():
                query_output = self.question_encoder(**inputs)
                embedding = query_output.last_hidden_state[:, 0, :].cpu().detach().numpy()

            return embedding
        
        
        # Step 1: Preprocess and index the corpus
    def build_faiss_index(self, target_corpus):
        logging.info("Building FAISS index...")
            
        # Initialize a FAISS index for L2 distance
        index = faiss.IndexFlatL2(self.embedding_dimension)

        # Process passages in chunks to avoid memory issues
        for i in range(0, len(target_corpus), self.batch_size):
            batch = target_corpus[i:i + self.batch_size]
            logging.info(f"Processing batch {i // self.batch_size + 1} of {len(target_corpus) // self.batch_size + 1}")
            embeddings = self.encode_passages(batch)
            index.add(embeddings)  

        logging.info("FAISS index built.")
        
        return index
                
    
    def build_candidate_embeddings(self, corpus):
        logging.info("Building candidate embeddings...")

        total_embeddings = []
        # Process passages in chunks to avoid memory issues
        ## Batch size always hard coded to 1
        for i in range(0, len(corpus)):
            logging.info(f"Processing batch {i + 1} of {len(corpus)  + 1}")
            embeddings = self.encode_passages([corpus[i]])
            total_embeddings.append(embeddings)

        logging.info(f"Length of original corpus is {len(corpus)}")
        
        return total_embeddings
    

    def retrieve_top_k(self, query, top_k, agent_name, current_faiss_index, target_corpus) -> list:
        """
        Retrieve top k papers for given input query from target corpus
        """     
        query_embedding = self.encode_query(query)  # Encode the query
        
        distances, indices = current_faiss_index.search(query_embedding, top_k)
        top_k_corpus = [(target_corpus[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
   
        return top_k_corpus    

    
    def Total_Merge_Results(self, args, organized_retrieved_top_corpus_per_agent) -> list:
        """
        Merge Each aspect-related results as a whole using Reciprocal Rank Fusion subsequent to chain of retrieval.
        """
        def compute_RRF(args, organized_retrieved_top_corpus_list) -> list:
            """
            Compute RRF between Organized Retrieved Results without differentiating the depth and origin of its retrieved result.
            """
            fused_scores = {}
            for top_k_corpus in organized_retrieved_top_corpus_list:
                for rank, corpus in enumerate(top_k_corpus):
                    fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)
                                
            sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
            parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
                        
            return parsed_retrieved_top_corpus
        
        per_agent_list = list(organized_retrieved_top_corpus_per_agent.keys())
        total_retrieved_result = []
        for agent_name in per_agent_list:
            if agent_name != "Original Retrieval":
                tree_structure_previous_memory = organized_retrieved_top_corpus_per_agent[agent_name]
                total_agent_depth_merged_results = []
                for iteration in range(0, self.total_iterative_retrieval_loop):
                    per_depth_saved_result_nodes = tree_structure_previous_memory[iteration]

                    for node in per_depth_saved_result_nodes:
                        total_agent_depth_merged_results.append(node.retrieved_result)

                logging.info(f"Number of Total corpus to merge for {agent_name} is {len(total_agent_depth_merged_results)}")
                per_agent_merged_result = compute_RRF(args, total_agent_depth_merged_results)

                total_retrieved_result.append(per_agent_merged_result)

        if self.include_original_retrieval == True:
            total_retrieved_result.append(organized_retrieved_top_corpus_per_agent["Original Retrieval"][0][0].retrieved_result)

        logging.info(f"Number of final corpus to merge is {len(total_retrieved_result)}")
        return total_retrieved_result


    
    def Recursive_Merge_Results(self, args, organized_retrieved_top_corpus_per_agent) -> list:
        """
        Recursively Merge Results for Iterative Tree Search subsequent to chain of retrieval
        """
        def compute_RRF(args, organized_tree_nodes_per_agent) -> list:
            """
            Compute RRF between Organized Nodes
            """
            fused_scores = {}
            organized_retrieved_top_corpus_list = []
            for node in organized_tree_nodes_per_agent:
                if len(node.retrieved_result) != 0:
                    logging.info("#####################")
                    logging.info(f"The current node name is {node.current_node_name}")
                    logging.info(f"The parent node of this current node is {node.parent}")
                    logging.info("#####################")
                organized_retrieved_top_corpus_list.append(node.retrieved_result)
                
            for top_k_corpus in organized_retrieved_top_corpus_list:
                for rank, corpus in enumerate(top_k_corpus):
                    fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)
                                
            sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
            parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
                        
            return parsed_retrieved_top_corpus
        
        def compute_RRF_between_parent_child(args, organized_retrieved_top_corpus_list) -> list:
            """
            Compute RRF between neighboring parent and child
            """
            fused_scores = {}
            for rank, corpus in enumerate(organized_retrieved_top_corpus_list[0]):
                fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)

            for rank, corpus in enumerate(organized_retrieved_top_corpus_list[1]):
                fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)

            sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
            parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
                        
            return parsed_retrieved_top_corpus

        per_agent_list = list(organized_retrieved_top_corpus_per_agent.keys())
        total_retrieved_result = []
        for agent_name in per_agent_list:
            if agent_name != "Original Retrieval":
                tree_structure_previous_memory = organized_retrieved_top_corpus_per_agent[agent_name]
                previous_depth_merged_results = []
                for iteration in range(0, self.total_iterative_retrieval_loop):
                    if iteration == 0:
                        previous_depth_merged_results.append([])
                        results_from_depth = tree_structure_previous_memory[self.total_iterative_retrieval_loop - 1 - iteration]
                        
                        number_of_results_from_current_depth = len(results_from_depth)
                        
                        logging.info(f"Number of results from current depth is {number_of_results_from_current_depth}")
                        logging.info(f"Number of total queries is {self.number_of_total_queries}")
                        for index in range(0, number_of_results_from_current_depth, self.number_of_total_queries):
                            organized_tree_nodes = results_from_depth[index:index + self.number_of_total_queries]
                            logging.info(f"Length of organized tree nodes is {len(organized_tree_nodes)}")
                            
                            ### Check whether they have same parents
                            current_parent_name = organized_tree_nodes[0].parent
                            for node in organized_tree_nodes:
                                parent_name = node.parent
                                if parent_name != current_parent_name:
                                    raise ValueError("Currently Tree Nodes have different parents")
                                
                            merged_siblings = compute_RRF(args, organized_tree_nodes)
                        
                            previous_depth_merged_results[iteration].append((current_parent_name, merged_siblings))

                    else:
                        previous_depth_merged_results.append([])
                        previous_merged_result = previous_depth_merged_results[iteration - 1]
                        results_from_current_depth = tree_structure_previous_memory[self.total_iterative_retrieval_loop - 1 - iteration]
                        index = 0
                        
                        for idx, tree_node in enumerate(results_from_current_depth):
                            agent_name = tree_node.current_node_name
                            top_k_corpus = tree_node.retrieved_result
                            if agent_name != "Original Retrieval":
                                organized_retrieved_top_corpus_parent_and_child = [top_k_corpus, previous_merged_result[index][1]]
                                if agent_name != previous_merged_result[index][0]:
                                    raise ValueError("Cannot merge results from different children")
                                
                                depth_wise_merged_result = compute_RRF_between_parent_child(args, organized_retrieved_top_corpus_parent_and_child)
                                
                                ## update the retrieved result from higher depth
                                tree_structure_previous_memory[self.total_iterative_retrieval_loop -1 - iteration][idx].retrieved_result = depth_wise_merged_result
                                
                                index = index + 1
                        
                        number_of_results_from_current_depth = len(tree_structure_previous_memory[self.total_iterative_retrieval_loop - 1 - iteration])
                        results_from_depth = tree_structure_previous_memory[self.total_iterative_retrieval_loop - 1 - iteration]
                        logging.info(f"Number of total queries is {self.number_of_total_queries}")
                        for index in range(0, number_of_results_from_current_depth, self.number_of_total_queries):
                            try:
                                organized_tree_nodes = results_from_depth[index:index + self.number_of_total_queries]
                            except:
                                organized_tree_nodes = results_from_depth[index:index + 1]
                            
                            logging.info(f"Length of organized tree nodes is {len(organized_tree_nodes)}")
                            current_parent_name = organized_tree_nodes[0].parent
                            for node in organized_tree_nodes:
                                parent_name = node.parent
                                if parent_name != current_parent_name:
                                    raise ValueError("Currently Tree Nodes have different parents")
                                
                            merged_siblings = compute_RRF(args, organized_tree_nodes)
                        
                            previous_depth_merged_results[iteration].append((current_parent_name, merged_siblings))
    
                ## check before appending the merged result
                if len(previous_depth_merged_results[self.total_iterative_retrieval_loop - 1]) != 1:
                    raise ValueError("Error with Merging Process")
                
                total_retrieved_result.append(previous_depth_merged_results[self.total_iterative_retrieval_loop - 1][0][1])
        
        if self.include_original_retrieval == True:
            total_retrieved_result.append(organized_retrieved_top_corpus_per_agent["Original Retrieval"][0][0].retrieved_result)

        logging.info(f"Number of final corpus to merge is {len(total_retrieved_result)}")
        return total_retrieved_result
    
    
    def SubTreeExplore(self, args, query_full_paper, query_abstract, parent_name, current_faiss_index, target_corpus, abstract_to_full_context_dict) -> tuple:
        """
        SubTreeExplore for Multi-Hop document Retrieval with depth-aware reinforced similiarity.
        """        
        if self.only_A2A == False:
            if args.use_abstract_for_query_optimization == True:
                optimized_queries = self.QueryOptimizer.forward_later_rounds(query_abstract, parent_name)
                
            formatted_query = []
            if self.include_original_retrieval == True:
                formatted_query.append(("Original Retrieval", query_abstract))

            for agent_name, query in optimized_queries:
                formatted_optimized_query = f"{query}"
                formatted_query.append((agent_name, formatted_optimized_query))
                
        elif self.only_A2A == True:
            if args.use_abstract_for_query_optimization == True:
                optimized_queries = self.QueryOptimizer.forward_later_rounds(query_abstract, parent_name, no_forward=True)
            
            formatted_query = []
            if self.include_original_retrieval == True:
                formatted_query.append((f"{parent_name}_Original Retrieval", query_abstract))

            for agent_name, query in optimized_queries:
                formatted_optimized_query = ""
                formatted_query.append((agent_name, formatted_optimized_query))

        #organized_retrieved_top_corpus_list = []
        organized_retrieved_top_corpus_per_agent = {}
        organized_retrieved_total_corpus_per_agent = {}
        
        for agent_name, current_query in formatted_query:
            if self.only_A2A == False:
                top_k_corpus = self.retrieve_top_k(current_query, args.top_k, agent_name, current_faiss_index, target_corpus)                
            elif self.only_A2A == True:
                if agent_name == f"{parent_name}_Original Retrieval" and current_query != "":
                    top_k_corpus = self.retrieve_top_k(current_query, args.top_k, "Original Retrieval", current_faiss_index, target_corpus)
                else:
                    top_k_corpus = []
                    
            #organized_retrieved_top_corpus_list.append(top_k_corpus)
            
            if self.only_A2A == False:
                current_corpus = []
                current_corpus_full_paper = []
                for rank, (corpus, score) in enumerate(top_k_corpus):
                    current_corpus.append(corpus)
                            
                    if agent_name != "Original Retrieval":
                        full_paper_content = abstract_to_full_context_dict[corpus]
                        current_corpus_full_paper.append(full_paper_content)
                    
                if agent_name != "Original Retrieval":        
                    organized_retrieved_top_corpus_per_agent[agent_name] = (current_corpus, current_corpus_full_paper)
                organized_retrieved_total_corpus_per_agent[agent_name] = current_corpus
                
            elif self.only_A2A == True:
                current_corpus = []
                current_corpus_full_paper = []
                for rank, (corpus, score) in enumerate(top_k_corpus):
                    current_corpus.append(corpus)
                            
                    full_paper_content = abstract_to_full_context_dict[corpus]
                    current_corpus_full_paper.append(full_paper_content)
                    
                            
                organized_retrieved_top_corpus_per_agent[agent_name] = (current_corpus, current_corpus_full_paper)
                organized_retrieved_total_corpus_per_agent[agent_name] = current_corpus
        
        if self.only_A2A == True:
            optimized_queries = formatted_query
            
        return organized_retrieved_total_corpus_per_agent, organized_retrieved_top_corpus_per_agent, optimized_queries


    def evaluate(self, args, test_file_directory, result_folder_path):
        """
        Evaluate Retrieval Performance on scientific papers submitted to recent venues, such as ICLR 2024, ICLR 2025.
        You can either use query optimizer or not. 
        """ 
        total_results = []
        for files in test_file_directory:
            if self.use_query_optimizer == False:
                with open(files, "r") as json_file:
                    evaluation_data = json.load(json_file)
                    
                formatted_query, formatted_correct_candidates = self.format_query_candidates(evaluation_data)
                raw_corpus = evaluation_data["Target_Corpus"]
                
                formatted_target_corpus, formatted_target_corpus_to_score, abstract_to_full_context_dict = self.format_corpus(raw_corpus)
                current_faiss_index = self.build_faiss_index(formatted_target_corpus)
                    
            elif self.use_query_optimizer == True:
                with open(files, "r") as json_file:
                    evaluation_data = json.load(json_file)
                
                query_title = evaluation_data["Query"]["title"]
                query_abstract = evaluation_data["Query"]["abstract"]
                original_formatted_query_title_abstract = f"Title: {query_title}\nAbstract: {query_abstract}"

                if args.use_abstract_for_query_optimization == True:
                    logging.info("Using abstract for query optimization")
                    evaluation_data, optimized_queries = self.QueryOptimizer.forward(files, evaluation_data)
                else:
                    raise TypeError("Must use abstracts for input query optimization")
                
                formatted_query, formatted_correct_candidates = self.format_query_candidates(evaluation_data)
                raw_corpus = evaluation_data["Target_Corpus"]
                
                formatted_target_corpus, formatted_target_corpus_to_score, abstract_to_full_context_dict = self.format_corpus(raw_corpus)
                current_faiss_index = self.build_faiss_index(formatted_target_corpus)
                            
            if self.use_query_optimizer == False:                   
                if self.iterative_retrieval == False:
                    parsed_retrieved_top_corpus = []
                    agent_name = None                    
                        
                    top_k_corpus = self.retrieve_top_k(formatted_query, args.top_k, agent_name, current_faiss_index, formatted_target_corpus)
                    organized_retrieved_top_corpus = {}
                    for idx, (corpus, score) in enumerate(top_k_corpus):
                        organized_retrieved_top_corpus[f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                        parsed_retrieved_top_corpus.append(corpus)

                else:
                    raise TypeError("Cannot use iterative retrieval without query optimizers on RELISH benchmark")

            elif self.use_query_optimizer == True and self.SubTreeSearch == False and self.iterative_retrieval == False:
                    organized_retrieved_top_corpus = {}
                    organized_retrieved_top_corpus_list = []
                    for agent_name, current_query in formatted_query:
                        organized_retrieved_top_corpus[agent_name] = {}
                        top_k_corpus = self.retrieve_top_k(current_query, args.top_k, agent_name, current_faiss_index, formatted_target_corpus)
                        organized_retrieved_top_corpus_list.append(top_k_corpus)
                                
                        for idx, (corpus, score) in enumerate(top_k_corpus):
                            organized_retrieved_top_corpus[agent_name][f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                        
                    fused_scores = {}
                    for top_k_corpus in organized_retrieved_top_corpus_list:
                        for rank, (corpus, score) in enumerate(top_k_corpus):
                            fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)
                                
                    sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
                    parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
                            
            
            elif self.use_query_optimizer == True and self.SubTreeSearch == True and self.iterative_retrieval == True:
                    organized_retrieved_top_corpus_list = []
                    organized_retrieved_top_corpus = {}
                    organized_retrieved_top_corpus_per_agent = {}
                    
                    previous_organized_retrieved_top_corpus_per_agent_cache = {}

                    for agent, current_query in formatted_query:
                        organized_retrieved_top_corpus_per_agent[agent] = []
                        for iteration in range(0, self.total_iterative_retrieval_loop):
                            organized_retrieved_top_corpus_per_agent[agent].append([])
                                
                        previous_organized_retrieved_top_corpus_per_agent_cache[agent] = []
                    
                    organized_retrieved_top_corpus_for_refinement = {}
                    formatted_query_dictionary = {}
                    for iteration in range(0, self.total_iterative_retrieval_loop):  
                        organized_retrieved_top_corpus_per_iteration_list = []
                        for agent_name, current_query in formatted_query:
                            if iteration == 0:
                                formatted_query_dictionary[agent_name] = current_query
                                top_k_corpus = self.retrieve_top_k(current_query, args.top_k, agent_name, current_faiss_index, formatted_target_corpus)
                                organized_retrieved_top_corpus_per_iteration_list.append((top_k_corpus, iteration))
                            
                                top_k_corpus_for_refinement = []
                                top_k_corpus_for_refinement_abstract_list = []
                                top_k_corpus_for_refinement_full_paper_list = []
                                formatted_top_k_corpus = []
                                
                                for idx, (corpus, score) in enumerate(top_k_corpus):
                                    top_k_corpus_for_refinement.append(corpus)
                                    formatted_top_k_corpus.append(corpus)

                                    if agent_name != "Original Retrieval":
                                        full_paper_content = abstract_to_full_context_dict[corpus]
                                        top_k_corpus_for_refinement_full_paper_list.append(full_paper_content)
        
                                top_k_corpus_for_refinement_abstract_list.append(corpus)
                                    
                                tree_node = Tree_Node(current_node_name=agent_name, parent_node_name=None, retrieved_result=formatted_top_k_corpus)
                                organized_retrieved_top_corpus_per_agent[agent_name][0].append(tree_node)
                                
                                if agent_name != "Original Retrieval":
                                    if args.use_aspect_aware_cache_for_selection == True:
                                        appropriate_corpus, appropriate_full_paper_corpus = self.Selector._find_nonoverlapping_corpus(top_k_corpus_for_refinement_abstract_list, top_k_corpus_for_refinement_full_paper_list, previous_organized_retrieved_top_corpus_per_agent_cache, agent_name, args.input_top_k_for_verifier)
                                    elif args.use_aspect_aware_cache_for_selection == False:
                                        logging.info("Not using Aspect Aware-Cache for Next Query Selection")
                                        appropriate_corpus, appropriate_full_paper_corpus = self.Selector._find_corpus_nearest(top_k_corpus_for_refinement_abstract_list, top_k_corpus_for_refinement_full_paper_list, previous_organized_retrieved_top_corpus_per_agent_cache, agent_name, args.input_top_k_for_verifier)
                                        
                                    for selected_corpus in appropriate_corpus:
                                        previous_organized_retrieved_top_corpus_per_agent_cache[agent_name].append(selected_corpus)
                                    organized_retrieved_top_corpus_for_refinement[agent_name] = (appropriate_corpus, appropriate_full_paper_corpus)
                        
                                                
                        if iteration < self.total_iterative_retrieval_loop - 1:
                            logging.info(f"Currently Refining Query for {iteration + 1} Time.")
                            if (iteration + 1 >= args.starting_iteration_of_A2A_only - 1) and args.starting_iteration_of_A2A_only > 0:
                                self.only_A2A = True
                            
                            ### Verifier Select Appropriate candidates well aligned with Each aspect
                            logging.info(f"{list(organized_retrieved_top_corpus_for_refinement.keys())}")
                            filtered_candidates_dict = self.Selector.forward_for_multirounds(organized_retrieved_top_corpus_for_refinement=organized_retrieved_top_corpus_for_refinement)
                                
                            newly_formatted_query = []
                            for parent_agent, current_query in formatted_query:
                                if self.only_A2A == False:
                                    if parent_agent != "Original Retrieval":
                                        abstract, full_paper = filtered_candidates_dict[parent_agent]
                                        retrieved_total_corpus_per_agent, retrieved_corpus_per_agent, optimized_queries = self.SubTreeExplore(args, full_paper, abstract, parent_agent, current_faiss_index, formatted_target_corpus, abstract_to_full_context_dict)

                                        retrieved_corpus_per_agent_list = list(retrieved_corpus_per_agent.keys())

                                        retrieved_total_corpus_per_agent_list = list(retrieved_total_corpus_per_agent.keys())
                                            
                                        logging.info(f"Number of retrieved total corpus per agent list is {len(retrieved_total_corpus_per_agent)}")
                                        for agent in retrieved_total_corpus_per_agent_list:
                                            retrieved_corpus = retrieved_total_corpus_per_agent[agent]
                                                    
                                            tree_node = Tree_Node(current_node_name=agent, parent_node_name=parent_agent, retrieved_result=retrieved_corpus)
                                                    
                                            parent_list = parent_agent.split("_")
                                            root_parent = parent_list[0]
                                            logging.info(f"Root Parent Agent is {root_parent}")
                                            organized_retrieved_top_corpus_per_agent[root_parent][iteration + 1].append(tree_node)
                                            
                                        for agent in retrieved_corpus_per_agent_list:
                                            if self.only_A2A == False:
                                                if agent != "Original Retrieval":
                                                    parent_list = agent.split("_")
                                                    root_parent = parent_list[0]
                                                    
                                                    if args.use_aspect_aware_cache_for_selection == True:
                                                        appropriate_corpus, appropriate_full_paper_corpus = self.Selector._find_nonoverlapping_corpus(retrieved_corpus_per_agent[agent][0], retrieved_corpus_per_agent[agent][1], previous_organized_retrieved_top_corpus_per_agent_cache, root_parent, args.input_top_k_for_verifier)
                                                    elif args.use_aspect_aware_cache_for_selection == False:
                                                        logging.info("Not using Aspect Aware-Cache for Next Query Selection")
                                                        appropriate_corpus, appropriate_full_paper_corpus = self.Selector._find_corpus_nearest(retrieved_corpus_per_agent[agent][0], retrieved_corpus_per_agent[agent][1], previous_organized_retrieved_top_corpus_per_agent_cache, root_parent, args.input_top_k_for_verifier)
                                                    
                                                    for selected_corpus in appropriate_corpus:
                                                        previous_organized_retrieved_top_corpus_per_agent_cache[root_parent].append(selected_corpus)
                                                    organized_retrieved_top_corpus_for_refinement[agent] = (appropriate_corpus, appropriate_full_paper_corpus)
                                                
                                            
                                        newly_formatted_query.extend(optimized_queries)
                                            
                                else:
                                    if parent_agent != "Original Retrieval":
                                        abstract, full_paper = filtered_candidates_dict[parent_agent]
                                        retrieved_total_corpus_per_agent, retrieved_corpus_per_agent, optimized_queries = self.SubTreeExplore(args, full_paper, abstract, parent_agent, current_faiss_index, formatted_target_corpus, abstract_to_full_context_dict)

                                        retrieved_corpus_per_agent_list = list(retrieved_corpus_per_agent.keys())

                                        retrieved_total_corpus_per_agent_list = list(retrieved_total_corpus_per_agent.keys())
                                                
                                        logging.info(f"Number of retrieved total corpus per agent list is {len(retrieved_total_corpus_per_agent)}")
                                        for agent in retrieved_total_corpus_per_agent_list:
                                            retrieved_corpus = retrieved_total_corpus_per_agent[agent]
                                                        
                                            tree_node = Tree_Node(current_node_name=agent, parent_node_name=parent_agent, retrieved_result=retrieved_corpus)
                                                        
                                            parent_list = parent_agent.split("_")
                                            root_parent = parent_list[0]
                                            logging.info(f"Root Parent Agent is {root_parent}")
                                            organized_retrieved_top_corpus_per_agent[root_parent][iteration + 1].append(tree_node)
                                                
                                        for agent in retrieved_corpus_per_agent_list:
                                            parent_list = agent.split("_")
                                            root_parent = parent_list[0]
                                            
                                            if args.use_aspect_aware_cache_for_selection == True:
                                                appropriate_corpus, appropriate_full_paper_corpus = self.Selector._find_nonoverlapping_corpus(retrieved_corpus_per_agent[agent][0], retrieved_corpus_per_agent[agent][1], previous_organized_retrieved_top_corpus_per_agent_cache, root_parent, args.input_top_k_for_verifier)
                                            elif args.use_aspect_aware_cache_for_selection == False:
                                                logging.info("Not using Aspect Aware-Cache for Next Query Selection")
                                                
                                                appropriate_corpus, appropriate_full_paper_corpus = self.Selector._find_corpus_nearest(retrieved_corpus_per_agent[agent][0], retrieved_corpus_per_agent[agent][1], previous_organized_retrieved_top_corpus_per_agent_cache, root_parent, args.input_top_k_for_verifier)
                                                
                                            for selected_corpus in appropriate_corpus:
                                                    previous_organized_retrieved_top_corpus_per_agent_cache[root_parent].append(selected_corpus)
                                                
                                            organized_retrieved_top_corpus_for_refinement[agent] = (appropriate_corpus, appropriate_full_paper_corpus)
                                                             
                                        newly_formatted_query.extend(optimized_queries)
                                        
                            formatted_query = newly_formatted_query
                            logging.info(f"Number of newly formatted queries are {len(formatted_query)}")
                                                            
                    if args.Recursive_Merge == True:
                        organized_retrieved_top_corpus_list = self.Recursive_Merge_Results(args, organized_retrieved_top_corpus_per_agent)
                    elif args.Recursive_Merge == False:
                        organized_retrieved_top_corpus_list = self.Total_Merge_Results(args, organized_retrieved_top_corpus_per_agent)

                    organized_retrieved_top_corpus = []
                    
                    fused_scores = {}
                    for top_k_corpus in organized_retrieved_top_corpus_list:
                        for rank, corpus in enumerate(top_k_corpus):
                            try:
                                fused_scores[corpus] = fused_scores.get(corpus, 0) + 1 / (args.hyperparameter_RRF + rank + 1)
                            except:
                                print(corpus)
                        
                    sorted_docs = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
                    parsed_retrieved_top_corpus = sorted_docs[:args.top_k]
 

            current_result = evaluate_retrieval_SciDocs(parsed_retrieved_top_corpus, formatted_correct_candidates, formatted_target_corpus_to_score, args.top_k, args.max_top_k)
            total_results.append(current_result)
                
            if self.use_query_optimizer == False:
                organized_results = {"query": {"id": f"{evaluation_data['Query']['doc_id']}", "content": f"{formatted_query}"}, 
                                     "Retrieved_Candidates": organized_retrieved_top_corpus, "Final_Ranked_Results": parsed_retrieved_top_corpus, "Correct_Candidates": formatted_correct_candidates, "Current Result": current_result}
            
            elif self.use_query_optimizer == True:
                queries = []
                for current_query in formatted_query:
                    query = {'agent': current_query[0], "content": current_query[1]}
                    queries.append(query)
                        
                organized_results = {"query": {"id": f"{evaluation_data['Query']['doc_id']}", "content": queries}, 
                                     "Retrieved_Candidates": organized_retrieved_top_corpus, "Final_Ranked_Results": parsed_retrieved_top_corpus, "Correct_Candidates": formatted_correct_candidates, "Current Result": current_result}
                    
            current_result_folder_path = f"{result_folder_path}/{evaluation_data['Query']['doc_id']}.json"
            with open(current_result_folder_path, "w") as json_file:
                json.dump(organized_results, json_file, indent=4)