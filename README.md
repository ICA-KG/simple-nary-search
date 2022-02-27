# Simple N-ary Search
N-ary Semantic Search In Memory

- [x] BM25 (Retrieve)
- [x] Semantic Search (Re-Rank)
- [x] Multi-Lingual Support

## Requirements
- python 3
- sentence-transformers [[homepage]](https://github.com/UKPLab/sentence-transformers)

## Example
```python
from sns import SNS

corpus = [ CORPUS ]
sns = SNS(corpus)
sns.get_top_k( QUERY , k=10, n=100)

```