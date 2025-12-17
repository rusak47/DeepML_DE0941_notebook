from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "dengcao/Qwen3-Reranker-0.6B",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "dengcao/Qwen3-Reranker-0.6B",
    trust_remote_code=True
)

def rerank_document(query: str, document: str) -> float:
    inputs = tokenizer(query, document, return_tensors="pt", truncation=True)
    score = torch.sigmoid(model(**inputs).logits)[0].item()
    return score
