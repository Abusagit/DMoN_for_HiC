Graph Clustering with Graph Neural Networks __Modified by F.Velikonivtsev__ ([Original project](https://github.com/google-research/google-research/tree/master/graph_embedding/dmon))
===============================

This is the implementation accompanying our paper, [Graph Clustering with Graph Neural Networks](https://arxiv.org/abs/2006.16904).

## Example usage 

This code will work on contact map woth features and graph structure encoded in `.npz` format, store output in `dmon.tsv`. It will be suitable for AMBER usage:


```python

python3 -m train --graph_path=contact_map.npz --dropout_rate=0.5 --out=data/dmon.tsv --header_file=ground_truth.tsv --sample_id=contact_map
```
