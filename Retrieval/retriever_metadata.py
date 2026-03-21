import torch
import faiss
import os
import sys
import numpy as np
import torch
from torch import nn as nn
from transformers import AutoTokenizer, AutoModel
from adapters import AutoAdapterModel
from sentence_transformers import SentenceTransformer
import json
import torch
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
# Add the root directory to Python's module search path
sys.path.append(root_dir)
from Retrieval.metrics import evaluate_retrieval

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your GPU setup.")

logging.basicConfig(level=logging.INFO)

class Retriever:
    def __init__(self, args, result_folder_path):
        """
        Our overall retriever that retrieves input paper from massive target paper corpus.
        """
        self.benchmark_directory = args.benchmark_directory
        self.embedding_model = args.embedding_model
        self.use_gpu = torch.cuda.is_available()
        self.embedding_model_max_length = None

        self.use_full_paper_as_corpus = args.use_full_paper_as_corpus
        self.use_full_paper_as_query = args.use_full_paper_as_query

        self.chunked_corpus_to_paper_dict = {}
        self.full_paper_corpus_to_paper_dict = {}
        
        self.remove_citations = args.remove_citations
        self.remove_mentions = args.remove_mentions
        
        self.author_data_incorporation = args.author_data_incorporation
 
        self.use_introduction_as_query = args.use_introduction_as_query
        self.use_introduction_as_corpus = args.use_introduction_as_corpus
        
        root_dir = os.path.dirname(args.corpus_directory)
        self.root_dir = os.path.join(root_dir, self.embedding_model)
        os.makedirs(self.root_dir, exist_ok=True)
        
        
        if self.use_full_paper_as_corpus == False:
            if args.author_data_incorporation == True:
                self.faiss_index_path = os.path.join(self.root_dir, f"DB_author_metadata.faiss")
        elif self.use_full_paper_as_corpus == True:
            if args.author_data_incorporation == True:
                self.faiss_index_path = os.path.join(self.root_dir, f"DB_author_metadata_full_paper.faiss")
            else:
                if args.use_introduction_as_corpus == True:
                    logging.info("Using introduction as corpus, mapping index path accordingly")
                    self.faiss_index_path = os.path.join(self.root_dir, f"DB_full_paper_introduction_only.faiss")
                else:
                    raise TypeError("Unable to map index path for full paper corpus")
        
        self.embedding_dimension = None
        self.batch_size = args.batch_size

        self.faiss_index = None

        num_devices = torch.cuda.device_count()
        
        device_list = []
        for device_number in range(num_devices):
            device_list.append(f"cuda:{device_number}")

        assert len(device_list) > 0, f"Expected at least 1 devices, but found {len(device_list)}"

        with open(args.corpus_directory, "r") as json_file:
            self.corpus = json.load(json_file)
        
        self.device = device_list[-1]
        self.cuda_embedding_model_device = device_list[-1]
        
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
            
    
    def initialize(self):
        self.format_corpus()
        os.makedirs(self.root_dir, exist_ok=True)
        self.faiss_index = self.build_faiss_index()
    
    
    def format_corpus(self):        
        if self.use_full_paper_as_corpus == True:
            print("Formatting corpus for full paper for single source")
            total_full_paper_corpus = []
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper['abstract']
                arxiv_full_paper_directory = paper["full_paper_directory"]

                if self.use_introduction_as_corpus == True:
                    paper_id = paper["id"]
                    full_paper_root = "Paper2PaperRetrievalBench/SciFullBench/Final_Dataset_w_citations_mentions_removed/Target_Corpus/arxiv_full_papers_intro_parsed"
                    arxiv_full_paper_directory = os.path.join(full_paper_root, f"{paper_id}.mmd")
                    with open(arxiv_full_paper_directory, "r") as file:
                        full_paper_content = file.read()

                else:
                    full_paper_root = "Paper2PaperRetrievalBench/SciFullBench"
                    arxiv_full_paper_directory = os.path.join(full_paper_root, arxiv_full_paper_directory)
                    with open(arxiv_full_paper_directory, "r") as file:
                        full_paper_content = file.read()
                
                author_information = paper['authors']
                
                if self.author_data_incorporation == True:    
                    full_paper_content_w_author_info = f"Author: {author_information}\nContent: {full_paper_content}"
                    formatted_ground_truth_candidate_w_author_info = f"Title: {paper_title}\nAuthor: {author_information}\nAbstract: {paper_abstract}"    
                    
                    try:
                        self.full_paper_corpus_to_paper_dict[full_paper_content_w_author_info]
                    except KeyError:
                        total_full_paper_corpus.append(full_paper_content_w_author_info)
                        self.full_paper_corpus_to_paper_dict[full_paper_content_w_author_info] = formatted_ground_truth_candidate_w_author_info

                else:
                    formatted_ground_truth_candidate = f"Title: {paper_title}\nAbstract: {paper_abstract}"    
                    try:
                        self.full_paper_corpus_to_paper_dict[full_paper_content]
                    except KeyError:
                        total_full_paper_corpus.append(full_paper_content)
                        self.full_paper_corpus_to_paper_dict[full_paper_content] = formatted_ground_truth_candidate

            self.corpus = total_full_paper_corpus

        elif self.use_full_paper_as_corpus == False:
            logging.info("Formatting corpus for abstracts without chunked or full content")
            formatted_total_corpus = []
            formatted_total_corpus_dict = {}
            for paper in self.corpus:
                paper_title = paper["title"]
                paper_abstract = paper["abstract"]
                author_information = paper['authors']
                
                if self.author_data_incorporation == True:    
                    formatted_ground_truth_candidate_w_author_info = f"Title: {paper_title}\nAuthor: {author_information}\nAbstract: {paper_abstract}"
                    
                    try:
                        formatted_total_corpus_dict[formatted_ground_truth_candidate_w_author_info]
                    except KeyError:   
                        formatted_total_corpus.append(formatted_ground_truth_candidate_w_author_info)
                        formatted_total_corpus_dict[formatted_ground_truth_candidate_w_author_info] = formatted_ground_truth_candidate_w_author_info
                    
            self.corpus = formatted_total_corpus
    
        
    def format_query_candidates(self, paper):
        if self.use_full_paper_as_query == False:
            paper_id = paper["id"]
            
            query_paper = paper["Query"]
            query_title = query_paper["title"]
            query_abstract = query_paper["abstract"]
            
            author_metadata_information_root_directory = "Paper2PaperRetrievalBench/SciFullBench/Final_Dataset_w_citations_mentions_removed/Benchmark/Final_MetaData_Info"
            author_metadata_information_directory = os.path.join(author_metadata_information_root_directory, f"{paper_id}.json")
            
            with open(author_metadata_information_directory, "r") as json_file:
                author_metadata = json.load(json_file)
            
            author_metadata_information = author_metadata["authors"]
            author_concatenated_information = ", ".join(author_metadata_information)
            
            if self.author_data_incorporation == True:
                formatted_query = f"Title: {query_title}\nAuthor: {author_concatenated_information}\nAbstract: {query_abstract}"

        elif self.use_full_paper_as_query == True:
            paper_id = paper["id"]
            paper_filename = f"{paper_id}.mmd"
                
            parent_dir = os.path.dirname(self.benchmark_directory)
            benchmark_root_dir = os.path.dirname(parent_dir)

            if self.use_introduction_as_query == True:
                root_folder_name = os.path.join(benchmark_root_dir, "Full_Papers_Intro_Parsed")

            else:
                if self.remove_citations == False and self.remove_mentions == False:
                    root_folder_name = os.path.join(benchmark_root_dir, "Full_Papers")
                
                if self.remove_citations == True and self.remove_mentions == True:
                    root_folder_name = os.path.join(benchmark_root_dir, "Full_Papers_remove_citations_True_remove_mentions_True")

            full_paper_path = os.path.join(root_folder_name, paper_filename)

            logging.info(f"{full_paper_path}")

            with open(full_paper_path, "r") as file:
                paper_content = file.read()

            if self.author_data_incorporation == True:
                    
                author_metadata_information_root_directory = "Paper2PaperRetrievalBench/SciFullBench/Final_Dataset_w_citations_mentions_removed/Benchmark/Final_MetaData_Info"
                author_metadata_information_directory = os.path.join(author_metadata_information_root_directory, f"{paper_id}.json")
                
                with open(author_metadata_information_directory, "r") as json_file:
                    author_metadata = json.load(json_file)

                author_metadata_information = author_metadata["authors"]
                author_concatenated_information = ", ".join(author_metadata_information)
                
                if self.author_data_incorporation == True:
                    formatted_query = f"Author: {author_concatenated_information}\nContent: {paper_content}"
            
            else:
                logging.info("Using full paper content without author information as query")
                formatted_query = paper_content
        
        formatted_candidates = []
        candidate_papers = paper["Candidate"]
        for candidate in candidate_papers:
            candidate_title = candidate["title"]
            candidate_abstract = candidate["abstract"]
            if self.author_data_incorporation == True:
                candidate_author_information = candidate["authors"]
                formatted_candidate = f"Title: {candidate_title}\nAuthor: {candidate_author_information}\nAbstract: {candidate_abstract}"
            else:
                formatted_candidate = f"Title: {candidate_title}\nAbstract: {candidate_abstract}"
            
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
    def build_faiss_index(self):
        if os.path.exists(self.faiss_index_path):
            logging.info(f"Loading existing FAISS index...{self.faiss_index_path}")
            index = faiss.read_index(self.faiss_index_path)
            
            return index

        logging.info("Building FAISS index...")
            
        # Initialize a FAISS index for Minimum L2 distance.
        index = faiss.IndexFlatL2(self.embedding_dimension)

        logging.info(f"Corpus size: {len(self.corpus)}")
        # Process passages in chunks to avoid memory issues
        for i in range(0, len(self.corpus), self.batch_size):
            batch = self.corpus[i:i + self.batch_size]
            logging.info(f"Processing batch {i // self.batch_size + 1} of {len(self.corpus) // self.batch_size + 1}")
            embeddings = self.encode_passages(batch)
            index.add(embeddings)  

        # Save the index to disk
        faiss.write_index(index, self.faiss_index_path)
        logging.info("FAISS index built and saved.")

        return index
    
    # Calculate similarities and retrieve top-k abstracts
    def retrieve_top_k(self, query, top_k):        
        query_embedding = self.encode_query(query)  # Encode the query
        if self.use_full_paper_as_corpus == False:
            logging.info(f"Retrieving without using full papers")
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            top_k_corpus = [(self.corpus[idx], distances[0][i]) for i, idx in enumerate(indices[0])]    
        elif self.use_full_paper_as_corpus == True:
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            top_k_corpus = [(self.full_paper_corpus_to_paper_dict[self.corpus[idx]], distances[0][i], self.corpus[idx]) for i, idx in enumerate(indices[0])]
        
        return top_k_corpus
    
    def evaluate(self, args, test_file_directory, result_folder_path):
        """
        Evaluate Retrieval Performance on scientific papers submitted to recent venues, such as ICLR 2024, ICLR 2025.
        You can either use query optimizer or not. 
        """             
        total_results = []
        for files in test_file_directory:
            with open(files, "r") as json_file:
                evaluation_data = json.load(json_file)

            formatted_query, formatted_correct_candidates = self.format_query_candidates(evaluation_data)
                    
            parsed_retrieved_top_corpus = []
            top_k_corpus = self.retrieve_top_k(formatted_query, args.top_k)
                    
            if self.use_full_paper_as_corpus == True:
                organized_retrieved_top_corpus = {}    
                for idx, (corpus, score, original_content) in enumerate(top_k_corpus):
                    organized_retrieved_top_corpus[f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                    parsed_retrieved_top_corpus.append(corpus)
                    
            elif self.use_full_paper_as_corpus == False:
                organized_retrieved_top_corpus = {}
                for idx, (corpus, score) in enumerate(top_k_corpus):
                    organized_retrieved_top_corpus[f"\nRank {idx + 1}"] = {"Score": f"Score: {score:.4f}", "Content": f"{corpus}"}
                    parsed_retrieved_top_corpus.append(corpus)

            current_result = evaluate_retrieval(parsed_retrieved_top_corpus, formatted_correct_candidates, args.top_k, args.max_top_k)
            total_results.append(current_result)

            organized_results = {"query": {"id": f"{evaluation_data['id']}", "content": f"{formatted_query}"}, 
                                     "Retrieved_Candidates": organized_retrieved_top_corpus, "Correct_Candidates": formatted_correct_candidates, "Current Result": current_result}
                
                    
            current_result_folder_path = f"{result_folder_path}/{evaluation_data['id']}.json"
            with open(current_result_folder_path, "w") as json_file:
                json.dump(organized_results, json_file, indent=4)
                