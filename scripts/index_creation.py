import os
import shutil
from git import Repo
from sentence_transformers import SentenceTransformer
import faiss
from os.path import isfile, join
from glob import glob

from fixed_token_chunker import FixedTokenChunker
import math

CHUNK_SIZE = 250
CHUNK_PART = 0.25

def clone_repo(github_url, path='git_rep'):
    if os.path.exists(path):
        return path
    Repo.clone_from(github_url, path)
    return path

def get_files(repo_path):
    files = [f for f in glob(f"{repo_path}/**/*", recursive=True) if isfile(f)]
    return files

def read_files(files, chunker):
    docs = []
    metadata = []
    num_chunks = []

    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()
                chunks = chunker.split_text(content)

                num_chunks.append(len(chunks)) 
                for i, chunk in enumerate(chunks):
                    docs.append(chunk)
                    metadata.append(f"{f}::chunk_{i}")
        except:
            continue
    return docs, metadata, num_chunks

def indexing(docs, metadata, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, convert_to_tensor=False)
    
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings, metadata

def build_index_from_github(github_url):
    repo_path = clone_repo(github_url)
    files = get_files(repo_path)

    chunker = FixedTokenChunker(chunk_size=CHUNK_SIZE, chunk_overlap=int(CHUNK_PART * CHUNK_SIZE), encoding_name="cl100k_base")

    docs, metadata, num_chunks = read_files(files, chunker)
    index, embeddings, metadata = indexing(docs, metadata)
    return index, embeddings, metadata, docs, num_chunks    