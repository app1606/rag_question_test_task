# RAG Question answering Test Task Solution

The notebook with the solution and reports may be found in this folder, `notebooks/RAG_questions_test_task.ipynb`. The work was done and ran in Colab, so I would recommend using this [link](https://colab.research.google.com/drive/1moIvQge-K-rRqcCVTjmNlXQ5fNruogZ_?usp=sharing) to check the results yourself. In order to do so, one has to add files 

```
scripts/base_chunker.py
scripts/fixed_token_chunker.py
data/escrcpy-commits-generated.json
```

to the Colab content folder. If you just want to run the code and check the results, use `notebooks/RAG_questions_examples.ipynb`, Colab link is [here](https://colab.research.google.com/drive/1dTeZZ42l6-cKNJhGrUGPKzkT_rZHB9nJ?usp=sharing). To run it in Colab, files  

```
scripts/index_search.py
scripts/index_creation.py
scripts/index_model_evaluation.py
```
have to be added to the content folder along with the files mentioned before. It takes around 12 minutes to run the whole notebook on CPU. In both cases it's enough to just run the notebook.

## Conclusions

I managed to implement the RAG pipeline and improve the quality of the baseline model. I've added Chunking and Reranker on top of the basic indexing pipeline, which led to the increase in quality. `Recall@10` grew from 0.5 to 0.57. It takes significantly more time to run the model with Reranker, probably, the Reranking model has to be fine-tuned to increase the quality, otherwise "just Chunking" method  is faster hence it's more applicable. 
