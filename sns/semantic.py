# -*- coding: utf-8 -*-
import torch
from sentence_transformers import SentenceTransformer, util


class SemanticSearch:
    def __init__(self, model_name_or_path='distiluse-base-multilingual-cased-v2', device=None):
        self.ss = SentenceTransformer(model_name_or_path=model_name_or_path, device=device)

    def tokenize(self, x):
        return self.ss.tokenize(x)

    def get_top_k(self, query, documents, k=10):
        d_embeddings = self.ss.encode(documents, convert_to_tensor=True)
        q_embedding = self.ss.encode(query, convert_to_tensor=True)

        cos_scores = util.cos_sim(q_embedding, d_embeddings)[0]
        top_results = torch.topk(cos_scores, k=k)

        return [i for i in top_results[1]]
