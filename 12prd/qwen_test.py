#https://apidog.com/blog/qwen-3-embedding-reranker-ollama/
import ollama
import numpy as np

from FlagEmbedding import BGEM3FlagModel
from pydantic import BaseModel, Field
from other.qwen_test2 import rerank_document

# --- Model Definitions ---
EMBEDDING_MODEL = 'gemma3:1b'
#RERANKER_MODEL = 'dengcao/Qwen3-Reranker-4B:Q5_K_M' # to heavy for gtx1650
#RERANKER_MODEL = 'dengcao/Qwen3-Reranker-0.6B:F16' # also struggling
RERANKER_MODEL = 'dengcao/Qwen3-Reranker-0.6B:Q8_0'

model_embeddings = BGEM3FlagModel(
    'BAAI/bge-m3',  # multilingual model from HF
    use_fp16=True,  # shorter embeddings
    return_sparse=False,
    return_colbert_vecs=False,
    # not just 1 vector for each sentence, but a matrix where each line represents different similarity features
    return_dense=True
)

def rerank_document_broken(query: str, document: str) -> float:
    """
    Uses the Qwen3 Reranker to score the relevance of a document to a query.
    Returns a score of 1.0 for 'Yes' and 0.0 for 'No'.
    """

    class RankerResponse(BaseModel):
        #chain_of_thought: str = Field(  # adding this to not thinking model increases accuracy in avg up to 30%
        #    ...,
        #    description='Explain step by step how you decided the fitting score for each document .'
        #)
        score: str = Field(
            ...,
            description="The ranking score for document in range 0.0-1.0"
        )

    prompt = f"""
    You are an expert relevance grader. Your task is to evaluate if the
    following document is relevant to the user's query.
    Please answer with a simple 'Yes' or 'No'.

    Query: {query}
    Document: {document}
    """
    prompt2 = f"""
        Query: {query}\n
        Document: {document}
        """
    try:
        response = ollama.generate(
            model=RERANKER_MODEL,
            #messages=[{'role': 'user', 'content': prompt2}],
            prompt=prompt2,
            options={
                'temperature': 0.0
                ,'num_predict': 15  # IMPORTANT <- maximum number of tokens the model is allowed to generate.
            } # For deterministic output
            #,format=RankerResponse.model_json_schema() # dont use format with rerankers
        )
        print(response)
        answer = response['message']['content'].strip().lower()
        if 'yes' in answer:
            return 1.0
        return 0.0
    except Exception as e:
        print(f"An error occurred during reranking: {e}")
        return 0.0


# --- 1. Corpus and Offline Embedding Generation ---
documents = [
    "Embedding models convert text into numerical vectors for semantic search.",
    "To install Linux with Ollama support you can use an official release.",
    "The Qwen3 series of models was developed by Alibaba Cloud.",
    "Ollama provides a simple command-line interface for running LLMs.",
    "A reranker model refines search results by calculating a precise relevance score.",
    "To install Ollama on Linux, you can use a curl command.",
]

# In a real application, you would store these embeddings in a vector database
corpus_embeddings = []
print("Generating embeddings for the document corpus...")
for doc in documents:
    #response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=doc)
    embedded = model_embeddings.encode(doc)
    #corpus_embeddings.append(response['embedding'])
    corpus_embeddings.append(embedded['dense_vecs'])
print("Embeddings generated.")

def cosine_similarity(v1, v2):
    """Calculates cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# --- 2. Online Retrieval and Reranking ---
user_query = "How do I install Ollama?"

# Embed the user's query
#query_embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=user_query)['embedding']
query_embedding = model_embeddings.encode(user_query)['dense_vecs']

# Perform initial retrieval (semantic search)
retrieval_scores = [cosine_similarity(query_embedding, emb) for emb in corpus_embeddings]
top_k_indices = np.argsort(retrieval_scores)[::-1][:3] # Get top 3 results

print("\n--- Initial Retrieval Results (before reranking) ---")
for i in top_k_indices:
    print(f"Score: {retrieval_scores[i]:.4f} - Document: {documents[i]}")

# --- 3. Rerank the top results ---
retrieved_docs = [documents[i] for i in top_k_indices]

print("\n--- Reranking the top results ---")
reranked_scores = [rerank_document(user_query, doc) for doc in retrieved_docs]

# Combine documents with their new scores and sort
reranked_results = sorted(zip(retrieved_docs, reranked_scores), key=lambda x: x[1], reverse=True)

print("\n--- Final Results (after reranking) ---")
for doc, score in reranked_results:
    print(f"Relevance Score: {score:.2f} - Document: {doc}")