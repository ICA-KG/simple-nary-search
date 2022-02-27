# -*- coding: utf-8 -*-
import logging

from sns.bm25 import BM25Okapi
from sns.semantic import SemanticSearch


class SNS:
    """
    Simple Nary Search
    """

    def __init__(self, corpus, preprocess=None, device=None):
        self.corpus_raw = corpus
        self.preprocess = preprocess
        logging.info('Initializing ...')

        logging.info('Preprocessing ...')
        if self.preprocess:
            corpus = [self.preprocess(c) for c in corpus]
        logging.info('Preprocessing OK!')

        logging.debug('SS ...')
        self.ss = SemanticSearch(device=device)
        logging.info('SS OK!')

        logging.info('BM25 ...')
        self.bm25 = BM25Okapi(corpus, self.ss.tokenize)
        logging.info('BM25 OK!')
        logging.info('Initializing OK!')
        self.corpus = corpus

    def get_top_k(self, query, k=10, n=100):

        if self.preprocess:
            query = self.preprocess(query)

        retrieve_ids = self.bm25.get_top_n(query, n)
        rerank_ids = self.ss.get_top_k(query, [self.corpus[i] for i in retrieve_ids], k)

        return [self.corpus_raw[retrieve_ids[i]] for i in rerank_ids], [self.corpus_raw[i] for i in retrieve_ids]
